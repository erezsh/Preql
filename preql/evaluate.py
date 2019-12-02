# Steps for evaluation
#     expand
#         expand names
#         expand function calls
#     resolve types
#         propagate types
#         verify correctness
#         adjust tree to accomodate type semantics
#     simplify (opt)
#         compute local operations (fold constants)
#     compile
#         generate sql for remote operations
#     execute
#         execute remote queries
#         simplify (compute) into the final result

from typing import List, Optional, Any
from contextlib import contextmanager

from .dispatchy import Dispatchy

from .exceptions import pql_NameNotFound, pql_TypeError
from .utils import safezip, dataclass, SafeDict
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
RawSql = sql.RawSql
Sql = sql.Sql

dy = Dispatchy()




class State:
    def __init__(self, db, fmt):
        self.db = db
        self.fmt = fmt

        self.ns = [_initial_namespace()]
        self.tick = 0

    def get_var(self, name):
        for scope in reversed(self.ns):
            if name in scope:
                return scope[name]

        meta = dict(
            line = name.line,
            column = name.column,
        )
        raise pql_NameNotFound(str(name), meta)

    def set_var(self, name, value):
        self.ns[-1][name] = value

    def get_all_vars(self):
        d = {}
        for scope in self.ns:
            d.update(scope) # Overwrite upper scopes
        return d

    def push_scope(self):
        self.ns.append({})

    def pop_scope(self):
        return self.ns.pop()


    def __copy__(self):
        s = State(self.db, self.fmt)
        s.ns = [dict(n) for n in self.ns]
        s.tick = self.tick
        return s

    @contextmanager
    def use_scope(self, scope: dict):
        x = len(self.ns)
        self.ns.append(scope)
        try:
            yield
        finally:
            self.ns.pop()
            assert x == len(self.ns)


def pql_limit(state: State, table: objects.TableInstance, length: objects.Instance):
    table = compile_remote(state, table)
    length = compile_remote(state, length)
    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], limit=length.code)
    return table.remake(code=code)

internal_funcs = {
    'limit': pql_limit
}

def _initial_namespace():
    ns = SafeDict({p.name: p for p in types.primitives_by_pytype.values()})
    ns.update({
        name: objects.InternalFunction(name, [
            objects.Param(name) for name, type_ in list(f.__annotations__.items())[1:]
        ], f) for name, f in internal_funcs.items()
    })
    return ns




@dy
def resolve(state: State, struct_def: ast.StructDef):
    members = {str(k):resolve(state, v) for k, v in struct_def.members}
    struct = types.StructType(struct_def.name, members)
    state.set_var(struct_def.name, struct)
    return struct

@dy
def resolve(state: State, table_def: ast.TableDef) -> types.TableType:
    t = types.TableType(table_def.name, {}, False)
    state.set_var(t.name, t)

    t.add_column(types.DatumColumnType("id", types.Int, primary_key=True, readonly=True))
    for c in table_def.columns:
        c = resolve(state, c)
        t.add_column(c)
    return t

@dy
def resolve(state: State, col_def: ast.ColumnDef) -> types.ColumnType:
    return types.make_column(col_def.name, resolve(state, col_def.type), col_def.query)

@dy
def resolve(state: State, type_: ast.Type) -> types.PqlType:
    return state.get_var(type_.name)




@dy
def compile_type_def(table: types.TableType) -> Sql:
    posts = []
    pks = []
    columns = []

    for c in table.flatten():
        if c.is_concrete:
            type_ = compile_type(c.type)
            columns.append( f"{c.name} {type_}" )
            if isinstance(c, types.RelationalColumnType):
                # TODO any column, using projection / get_attr
                s = f"FOREIGN KEY({c.name}) REFERENCES {c.type.name}(id)"
                posts.append(s)
            if c.primary_key:
                pks.append(c.name)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    if table.temporary:
        return RawSql(types.null, f"CREATE TEMPORARY TABLE {table.name} (" + ", ".join(columns + posts) + ")")
    else:
        return RawSql(types.null, f"CREATE TABLE if not exists {table.name} (" + ", ".join(columns + posts) + ")")

@dy
def compile_type(type: types.TableType):
    return type.name

@dy
def compile_type(type: types.Primitive):
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'date': "TIMESTAMP",
    }[type.name]
    if not type.nullable:
        s += " NOT NULL"
    return s




@dy
def _execute(state: State, struct_def: ast.StructDef):
    resolve(state, struct_def)

@dy
def _execute(state: State, table_def: ast.TableDef):
    # Create type and a corresponding table in the database
    t = resolve(state, table_def)
    sql = compile_type_def(t)
    state.db.query(sql)

@dy
def _execute(state: State, var_def: ast.VarDef):
    res = simplify(state, var_def.value)
    state.set_var(var_def.name, res)

@dy
def _execute(state: State, func_def: ast.FuncDef):
    # res = simplify(state, func_def.value)
    func = func_def.userfunc
    assert isinstance(func, objects.UserFunction)
    state.set_var(func.name, func)

@dy
def _execute(state: State, p: ast.Print):
    res = evaluate(state, p.value)
    print(res)


def execute(state, stmt):
    if isinstance(stmt, ast.Statement):
        return _execute(state, stmt)
    return evaluate(state, stmt)




# TODO Is simplify even helpful? Why not just compile everything?
#      evaluate() already compiles anyway, so what's the big deal?
#      an "optimization" for local tree folding can be added later,
#      and also prevent compiling lists (which simplify doesn't prevent)

@dy
def simplify(state: State, n: ast.Name):
    # Don't recurse simplify, to allow granular dereferences
    # The assumption is that the stored variable is already simplified
    return state.get_var(n.name)

@dy
def simplify(state: State, c: ast.Const):
    return c

# @dy
# def simplify(state: State, a: ast.Attr):
#     print("@@@@@", a)
#     return a

@dy
def simplify(state: State, funccall: ast.FuncCall):
    func = simplify(state, funccall.func)

    matched = func.match_params(funccall.args)
    args = [(p, simplify(state, a)) for p, a in matched]
    if isinstance(func, objects.UserFunction):
        with state.use_scope({p.name:a for p,a in args}):
            r = simplify(state, func.expr)
            return r
    else:
        return func.func(state, *[v for k,v in args])




@dy
def simplify(state: State, c: ast.Arith):
    return ast.Arith(c.op, simplify_list(state, c.args))

@dy
def simplify(state: State, c: objects.List_):
    return objects.List_(simplify_list(state, c.elems))

@dy
def simplify(state: State, c: ast.Compare):
    return ast.Compare(c.op, simplify_list(state, c.args))

@dy
def simplify(state: State, c: ast.Selection):
    # TODO: merge nested selection
    # table = simplify(state, c.table)
    # return ast.Selection(table, c.conds)
    return compile_remote(state, c)

@dy
def simplify(state: State, p: ast.Projection):
    # TODO: unite nested projection
    # table = simplify(state, p.table)
    # return ast.Projection(table, p.fields, p.groupby, p.agg_fields)
    return compile_remote(state, p)


# @dy
# def simplify(state: State, attr: ast.Attr):
#     print("##", attr.expr)
#     expr = simplify(state, attr.expr)
#     print("##", expr)
#     return ast.Attr(expr, attr.name)

def simplify_list(state, x):
    return [simplify(state, e) for e in x]


@dataclass
class TableConstructor(objects.Function):
    params: List[Any]
    param_collector: Optional[objects.Param] = None
    name = 'new'

@dy
def simplify(state: State, new: ast.New):
    # XXX This function has side-effects.
    # Perhaps it belongs in resolve, rather than simplify?
    table = state.get_var(new.type)
    assert isinstance(table, types.TableType)
    cons = TableConstructor(list(table.params()))
    matched = cons.match_params(new.args)

    destructured_pairs = []
    for k, v in matched:
        if isinstance(k.type, types.StructType):
            v = evaluate(state, v)
            v = [x['value'] for x in v] # does it matter if value? Any 1-column table should be fine
            for k2, v2 in safezip(k.flatten(), v):
                destructured_pairs.append((k2, v2))
        else:
            v = simplify(state, v)
            destructured_pairs.append((k, v.value))

    keys = [k.name for (k,_) in destructured_pairs]
    values = [sql_repr(v) for (_,v) in destructured_pairs]
    # sql = RawSql(f"INSERT INTO {table.name} ($keys_str) VALUES ($values_str)")
    q = sql.Insert(types.null, sql.TableName(table, table.name), keys, values)

    state.db.query(q)
    rowid = state.db.query(sql.LastRowId())
    return ast.Const(types.Int, rowid)   # Todo Row reference / query



def sql_repr(x):
    if x is None:
        return sql.null

    t = types.primitives_by_pytype[type(x)]
    return sql.Primitive(t, repr(x))


@dy # Could happen because sometimes simplify(...) calls compile_remote
def compile_remote(state: State, i: objects.Instance):
    return i
@dy
def compile_remote(state: State, i: objects.TableInstance):
    return i
@dy
def compile_remote(state: State, f: ast.FuncCall):
    return compile_remote(state, simplify(state, f))


@dy
def compile_remote(state: State, c: ast.Const):
    return objects.Instance.make(sql_repr(c.value), c.type, [])

@dy
def compile_remote(state: State, n: ast.Name):
    return compile_remote(state, simplify(state, n))

@dy
def compile_remote(state: State, lst: objects.List_):
    # TODO generate (a,b,c) syntax for IN operations, with its own type
    # sql = "(" * join([e.code.text for e in objs], ",") * ")"
    # type = length(objs)>0 ? objs[1].type : nothing
    # return Instance(Sql(sql), ArrayType(type, false))
    # Or just evaluate?

    elems = [compile_remote(state, e) for e in lst.elems]

    elem_type = elems[0].type if elems else types.PqlType
    table_type = types.ListType(elem_type)
    # table_type = types.TableType("array_%s" % elem_type.name, {}, temporary=True)   # Not temporary: ephemeral
    # table_type.add_column(types.DatumColumnType("value", elem_type))

    code = sql.TableArith(table_type, 'UNION ALL', [ sql.SelectValue(e.type, e.code) for e in elems ])
    inst = instanciate_table(state, table_type, code, elems)
    return inst
    # return add_as_subquery(state, inst)

@dy
def compile_remote(state: State, t: types.TableType):
    i = instanciate_table(state, t, sql.TableName(t, t.name), [])
    return i
    # return add_as_subquery(state, i)

@dy
def compile_remote(state: State, sel: ast.Selection):
    table = compile_remote(state, sel.table)

    with state.use_scope(table.columns):
        conds = compile_remote(state, sel.conds)

    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], conds=[c.code for c in conds])
    # inst = instanciate_table(state, table.type, code, [table] + conds)
    inst = objects.TableInstance.make(code, table.type, [table] + conds, table.columns)
    return inst
    # return add_as_subquery(state, inst)


def _ensure_col_instance(i):
    if isinstance(i, objects.ColumnInstance):
        return i
    elif isinstance(i, objects.Instance):
        if isinstance(i.type, types.Primitive):
            return objects.make_column_instance(i.code, types.make_column(i.type, '_anonymous_instance'))
    assert False


def guess_field_name(f):
    if isinstance(f, ast.Attr):
        return guess_field_name(f.expr) + "." + f.name
    elif isinstance(f, ast.Name):
        return str(f.name)
    elif isinstance(f, ast.Projection): # a restructre within a projection
        return guess_field_name(f.table)

    assert False, f


def _process_fields(state: State, fields):
    "Returns {var_name: (col_instance, sql_alias)}"
    processed_fields = {}
    for f in fields:
        if f.name in processed_fields:
            raise pql_TypeError(f"Field '{f.name}' was already used in this projection")

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name
        sql_friendly_name = name.replace(".", "_")
        unique_name = name
        i = 1
        while unique_name in processed_fields:
            i += 1
            unique_name = name + str(i)

        v = compile_remote(state, f.value)

        # TODO move to ColInstance
        if isinstance(v, Aggregated):
            expr = _ensure_col_instance(v.expr)
            list_type = types.ListType(expr.type.type)
            col_type = types.make_column(get_alias(state, "list"), list_type)
            v = objects.make_column_instance(sql.MakeArray(list_type, expr.code), col_type, [expr])
        elif isinstance(v, objects.TableInstance):
            t = types.make_column(name, types.StructType(v.name, v.members))
            v = objects.StructColumnInstance(v.code, t, v.subqueries, v.columns)
        v = _ensure_col_instance(v)
        processed_fields[unique_name] = v, get_alias(state, sql_friendly_name)   # TODO Don't create new alias for fields that don't change?

    return list(processed_fields.items())

@dataclass
class Aggregated(ast.Ast):
    expr: types.PqlObject

    def get_attr(self, name):
        return Aggregated(self.expr.get_attr(name))

@dy
def compile_remote(state: State, proj: ast.Projection):
    table = compile_remote(state, proj.table)

    columns = table.members if isinstance(table, objects.StructColumnInstance) else table.columns

    with state.use_scope(columns):
        fields = _process_fields(state, proj.fields)

    with state.use_scope({n:Aggregated(c) for n,c in columns.items()}):
        agg_fields = _process_fields(state, proj.agg_fields)

    if isinstance(table, objects.StructColumnInstance):
        members = {name: inst.type for name, (inst, _a) in fields + agg_fields}
        struct_type = types.StructType(get_alias(state, table.type.name + "_proj"), members)
        struct_col_type = types.make_column("<this is meaningless?>", struct_type)
        return objects.make_column_instance(table.code, struct_col_type, [table])

    # Make new type
    all_aliases = []
    new_table_type = types.TableType(get_alias(state, table.type.name + "_proj"), {}, True)
    for name, (remote_col, alias) in fields + agg_fields:
        new_col_type = remote_col.type.remake(name=name)
        new_table_type.add_column(new_col_type)
        ci = objects.make_column_instance(sql.Name(new_col_type.type, alias), new_col_type, [remote_col])
        all_aliases.append((remote_col, ci))

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in all_aliases
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    groupby = []
    if proj.groupby and fields:
        groupby = [sql.Name(rc.type.type, alias) for _n, (rc, alias) in fields]

    code = sql.Select(new_table_type, table.code, sql_fields, group_by=groupby)

    # Make Instance
    columns = {new.type.name:new for old, new in all_aliases}
    inst = objects.TableInstance.make(code, new_table_type, [table], columns)

    return inst
    # return add_as_subquery(state, inst)


@dy
def compile_remote(state: State, lst: list):
    return [compile_remote(state, e) for e in lst]

@dy
def compile_remote(state: State, cmp: ast.Compare):
    op = {
        '==': '='
    }.get(cmp.op, cmp.op)
    code = sql.Compare(types.Bool, op, [e.code for e in compile_remote(state, cmp.args)])
    return objects.Instance.make(code, types.Bool, [])

@dy
def compile_remote(state: State, attr: ast.Attr):
    inst = compile_remote(state, attr.expr)
    return inst.get_attr(attr.name)

@dy
def compile_remote(state: State, obj: objects.StructColumnInstance):
    return obj
@dy
def compile_remote(state: State, obj: objects.DatumColumnInstance):
    return obj
@dy
def compile_remote(state: State, obj: Aggregated):
    return obj



def add_as_subquery(state: State, inst: objects.Instance):
    name = get_alias(state, inst)
    if isinstance(inst, objects.TableInstance):
        new_inst = objects.TableInstance(sql.TableName(inst.code.type, name), inst.type, inst.subqueries.update({name: inst.code}), inst.columns)
    else:
        new_inst = objects.Instance(sql.Name(inst.code.type, name), inst.type, inst.subqueries.update({name: inst.code}))
    return new_inst

def evaluate(state, obj):
    obj = simplify(state, obj)
    if isinstance(obj, objects.Function):   # TODO base class on uncompilable?
        return obj

    code = compile_remote(state, obj)
    res = localize(state, code)
    # if isinstance(obj, objects.List_):
    #     assert all(len(x)==1 for x in res)
    #     return [x[0] for x in res]
    return res



def localize(state, inst):
    res = state.db.query(inst.code, inst.subqueries)
    return res


def get_alias(state: State, obj):
    if isinstance(obj, objects.TableInstance):
        return get_alias(state, obj.type.name)

    state.tick += 1
    return obj + str(state.tick)

def instanciate_column(state: State, c: types.ColumnType):
    return objects.make_column_instance(RawSql(c.type, get_alias(state, c.name)), c)

def instanciate_table(state: State, t: types.TableType, source: Sql, instances):
    columns = {name: instanciate_column(state, c) for name, c in t.columns.items()}

    aliases = [
        sql.ColumnAlias.make(sql.Name(dinst.type, dinst.type.name), dinst.code)
        for inst in columns.values()
        for dinst in inst.flatten()
    ]

    code = sql.Select(t, source, aliases)

    return objects.TableInstance(code, t, objects.merge_subqueries(instances), columns)
