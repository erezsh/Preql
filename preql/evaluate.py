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

from .exceptions import pql_NameNotFound, pql_TypeError, pql_JoinError
from .utils import safezip, dataclass, SafeDict, listgen
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

def _apply_sql_func(state, obj: ast.Expr, table_func, field_func):
    obj = compile_remote(state, obj)
    if isinstance(obj, objects.TableInstance):
        code = table_func(obj.type, obj.code)
    else:
        assert isinstance(obj, Aggregated)
        obj = obj.expr
        assert isinstance(obj, objects.ColumnInstance), obj
        code = field_func(types.Int, obj.code)

    return objects.Instance.make(code, types.Int, [obj])

def pql_count(state: State, obj: ast.Expr):
    return _apply_sql_func(state, obj, sql.CountTable, lambda t,c: sql.FieldFunc(t, 'count', c))

def pql_sum(state: State, obj: ast.Expr):
    return _apply_sql_func(state, obj, None, lambda t,c: sql.FieldFunc(t, 'sum', c))


def pql_enum(state: State, table: ast.Expr):
    index_name = "index"

    table = compile_remote(state, table)

    new_table_type = types.TableType(get_alias(state, "enum"), {}, True)
    new_table_type.add_column(types.make_column(index_name, types.Int))
    for c in table.type.columns.values():
        new_table_type.add_column(c)

    # Added to sqlite3 in 3.25.0: https://www.sqlite.org/windowfunctions.html
    index_code = sql.RawSql(types.Int, "row_number() over ()")
    values = [index_code] + [c.code for c in table.flatten()]

    return instanciate_table(state, new_table_type, table.code, [table], values=values)

def pql_temptable(state: State, expr: ast.Expr):
    expr = compile_remote(state, expr)
    assert isinstance(expr, objects.TableInstance)
    name = get_alias(state, "temp_" + expr.type.name)
    table = types.TableType(name, expr.type.columns, temporary=True)
    state.db.query(compile_type_def(table))
    state.db.query(sql.Insert(types.null, name, expr.code))
    return table




def sql_bin_op(state, op, table1, table2, name):
    t1 = compile_remote(state, table1)
    t2 = compile_remote(state, table2)
    # TODO make sure both table types are compatiable
    l1 = len(t1.type.flatten())
    l2 = len(t2.type.flatten())
    if l1 != l2:
        raise pql_TypeError(f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    code = sql.TableArith(t1.type, op, [t1.code, t2.code])
    return objects.TableInstance.make(code, t1.type, [t1, t2], t1.columns)

def pql_intersect(state, t1, t2):
    return sql_bin_op(state, "INTERSECT", t1, t2, "intersect")

def pql_substract(state, t1, t2):
    return sql_bin_op(state, "EXCEPT", t1, t2, "substract")

def pql_union(state, t1, t2):
    return sql_bin_op(state, "UNION", t1, t2, "union")

def pql_concat(state, t1, t2):
    return sql_bin_op(state, "UNION ALL", t1, t2, "concatenate")


def _join(state: State, join: str, exprs: dict, joinall=False):
    assert len(exprs) == 2
    exprs = {name: compile_remote(state, value) for name,value in exprs.items()}
    assert all(isinstance(x, objects.Instance) for x in exprs.values())

    (a,b) = exprs.values()

    if joinall:
        tables = (a,b)
    else:
        if isinstance(a, objects.ColumnInstanceWithTable) and isinstance(b, objects.ColumnInstanceWithTable):
            cols = a, b
        else:
            assert isinstance(a, objects.TableInstance) and isinstance(b, objects.TableInstance)    # TODO better error message (TypeError?)
            cols = _auto_join(state, join, a, b)
        tables = [c.table for c in cols]

    col_types = {name: types.make_column(name, types.StructType(name, {n:c.type.type for n, c in table.columns.items()}))
                for name, table in safezip(exprs, tables)}
    table_type = types.TableType(get_alias(state, "joinall" if joinall else "join"), col_types, False)

    conds = [] if joinall else [sql.Compare(types.Bool, '==', [cols[0].code, cols[1].code])]
    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    columns = dict(safezip(exprs, tables))
    return objects.TableInstance.make(code, table_type, [a,b], columns)

def pql_join(state, tables):
    return _join(state, "JOIN", tables)
def pql_leftjoin(state, tables):
    return _join(state, "LEFT JOIN", tables)

def _auto_join(state, join, ta, tb):
    refs1 = _find_table_reference(ta, tb)
    refs2 = _find_table_reference(tb, ta)
    auto_join_count = len(refs1) + len(refs2)
    if auto_join_count < 1:
        raise pql_JoinError("Cannot auto-join: No plausible relations found")
    elif auto_join_count > 1:   # Ambiguity in auto join resolution
        raise pql_JoinError("Cannot auto-join: Several plausible relations found")

    if len(refs1) == 1:
        dst, src = refs1[0]
    elif len(refs2) == 1:
        src, dst = refs2[0]
    else:
        assert False

    return src, dst


@listgen
def _find_table_reference(t1, t2):
    # XXX TODO need to check TableType too (owner)?
    for c in t1.columns.values():
        if isinstance(c.type, types.RelationalColumnType):
            rel = c.type.type
            if rel == t2.type:
                # TODO depends on the query
                yield (objects.ColumnInstanceWithTable(t2.get_attr('id'), t2), objects.ColumnInstanceWithTable(c, t1))


internal_funcs = {
    'limit': pql_limit,
    'count': pql_count,
    'sum': pql_sum,
    'enum': pql_enum,
    'temptable': pql_temptable,
}
joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param('tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param('tables')),
}

def _initial_namespace():
    ns = SafeDict({p.name: p for p in types.primitives_by_pytype.values()})
    ns.update({
        name: objects.InternalFunction(name, [
            objects.Param(name) for name, type_ in list(f.__annotations__.items())[1:]
        ], f) for name, f in internal_funcs.items()
    })
    ns.update(joins)
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
    # args = [(p, simplify(state, a)) for p, a in matched]
    args = matched
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
# def simplify(state: State, d: dict):
#     return {name: simplify(state, v) for name, v in d.items()}

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
            for k2, v2 in safezip(k.flatten(), v):
                destructured_pairs.append((k2, v2))
        else:
            v = simplify(state, v)
            destructured_pairs.append((k, v.value))

    keys = [k.name for (k,_) in destructured_pairs]
    values = [sql_repr(v) for (_,v) in destructured_pairs]
    # sql = RawSql(f"INSERT INTO {table.name} ($keys_str) VALUES ($values_str)")
    q = sql.InsertConsts(types.null, sql.TableName(table, table.name), keys, values)

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

    # TODO use values + subqueries instead of union all (better performance)
    # e.g. with list(value) as (values(1),(2),(3)) select value from list;
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
            return objects.make_column_instance(i.code, types.make_column('_anonymous_instance', i.type))

    assert False, i


def guess_field_name(f):
    if isinstance(f, ast.Attr):
        return guess_field_name(f.expr) + "." + f.name
    elif isinstance(f, ast.Name):
        return str(f.name)
    elif isinstance(f, ast.Projection): # a restructre within a projection
        return guess_field_name(f.table)
    elif isinstance(f, ast.FuncCall):
        return guess_field_name(f.func)
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

        if isinstance(f.value, ast.Name):   # No modification in projection
            processed_fields[unique_name] = v, v.code.text
        else:
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
    for name, (remote_col, sql_alias) in fields + agg_fields:
        new_col_type = remote_col.type.remake(name=name)
        new_table_type.add_column(new_col_type)
        ci = objects.make_column_instance(sql.Name(new_col_type.type, sql_alias), new_col_type, [remote_col])
        all_aliases.append((remote_col, ci))

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in all_aliases
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    groupby = []
    if proj.groupby and fields:
        groupby = [sql.Name(rc.type.type, sql_alias) for _n, (rc, sql_alias) in fields]

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
def compile_remote(state: State, c: ast.Contains):
    args = compile_remote(state, c.args)
    code = sql.Contains(types.Bool, c.op, [a.code for a in args])
    return objects.make_instance(code, types.Bool, args)

@dy
def compile_remote(state: State, arith: ast.Arith):
    args = compile_remote(state, arith.args)

    assert all(a.code.type==args[0].code.type for a in args), args
    # TODO check instance type? Right now ColumnInstance & ColumnType make it awkward

    if isinstance(args[0], objects.TableInstance):
        assert isinstance(args[1], objects.TableInstance)
        # TODO validate types

        ops = {
            "+": pql_concat,
            "&": pql_intersect,
            "|": pql_union,
            "-": pql_substract,
        }
        return ops[arith.op](state, *args)

    # TODO validate all args are compatiable
    # return Instance(Sql(join([a.code.text for a in args], arith.op)), args[1].type, args)
    code = sql.Arith(args[0].type, arith.op, [a.code for a in args])
    return objects.make_instance(code, args[0].type, [])

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

    inst = compile_remote(state, obj)
    res = localize(state, inst)
    # if isinstance(obj, objects.List_):
    #     assert all(len(x)==1 for x in res)
    #     return [x[0] for x in res]
    if isinstance(inst.type, types.ListType):
        assert all(len(e)==1 for e in res)
        return [e['value'] for e in res]

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

def instanciate_table(state: State, t: types.TableType, source: Sql, instances, values=None):
    columns = {name: instanciate_column(state, c) for name, c in t.columns.items()}

    atoms = [atom
                for inst in columns.values()
                for atom in inst.flatten()
            ]

    if values is None:
        values = [sql.Name(atom.type, atom.type.name) for atom in atoms]    # The column value

    aliases = [ sql.ColumnAlias.make(value, atom.code) for value, atom in safezip(values, atoms) ]

    code = sql.Select(t, source, aliases)

    return objects.TableInstance(code, t, objects.merge_subqueries(instances), columns)
