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
from .dispatchy import Dispatchy

from .exceptions import pql_NameNotFound
from .utils import safezip, dataclass
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
RawSql = sql.RawSql
Sql = sql.Sql

dy = Dispatchy()

def _initial_namespace():
    return {
        'int': types.Int,
        'str': types.String,
    }

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

        raise pql_NameNotFound(name)

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




@dy
def resolve(state: State, struct_def: ast.StructDef):
    members = {k:resolve(state, v) for k, v in struct_def.members}
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
def compile_type(table: types.TableType) -> Sql:
    posts = []
    pks = []
    columns = []

    for c in table.flatten():
        if c.is_concrete:
            type_ = compile_type(c.type)
            columns.append( f"{c.name} {type_}" )
            if isinstance(c, types.RelationalColumnType):
                s = f"FOREIGN KEY({c.name}) REFERENCES {fk.table.name}({fk.column.name})"
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
def execute(state: State, struct_def: ast.StructDef):
    resolve(state, struct_def)

@dy
def execute(state: State, table_def: ast.TableDef):
    # Create type and a corresponding table in the database
    t = resolve(state, table_def)
    sql = compile_type(t)
    state.db.query(sql)

@dy
def execute(state: State, var_def: ast.VarDef):
    res = simplify(state, var_def.value)
    state.set_var(var_def.name, res)

@dy
def execute(state: State, p: ast.Print):
    res = evaluate(state, p.value)
    print(res)




@dy
def simplify(state: State, n: ast.Name):
    # Don't recurse simplify, to allow granular dereferences
    # The assumption is that the stored variable is already simplified
    return state.get_var(n.name)

@dy
def simplify(state: State, c: ast.Const):
    return c

@dy
def simplify(state: State, c: ast.Arith):
    return ast.Arith(c.op, simplify_list(state, c.args))

@dy
def simplify(state: State, c: objects.List_):
    return objects.List_(simplify_list(state, c.elems))

@dy
def simplify(state: State, c: ast.Compare):
    return ast.Compare(c.op, simplify_list(state, c.args))

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
            destructured_pairs.append((k, v))

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
    return RawSql(t, repr(x))


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
    return add_as_subquery(state, inst)

@dy
def compile_remote(state: State, t: types.TableType):
    i = instanciate_table(state, t, sql.TableName(t, t.name), [])
    return add_as_subquery(state, i)


def add_as_subquery(state: State, inst: objects.Instance):
    name = get_alias(state, inst)
    new_inst = objects.Instance(sql.Name(inst.code.type, name), inst.type, inst.subqueries.update({name: inst.code}))
    return new_inst

def evaluate(state, obj):
    obj = simplify(state, obj)
    code = compile_remote(state, obj)
    res = localize(state, code)
    # if isinstance(obj, objects.List_):
    #     assert all(len(x)==1 for x in res)
    #     return [x[0] for x in res]
    return res



def localize(state, inst):
    res = state.db.query(inst.code, inst.subqueries)
    return res


@dy
def get_alias(state: State, s: str):
    state.tick += 1
    return s + str(state.tick)

@dy
def get_alias(state: State, ti: objects.TableInstance):
    return get_alias(state, ti.type.name)

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
