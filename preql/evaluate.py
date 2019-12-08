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

from .utils import safezip, dataclass
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
RawSql = sql.RawSql
Sql = sql.Sql

from .interp_common import State, dy, get_alias
from .compiler import compile_remote, compile_type_def, sql_repr



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
    inst = evaluate(state, p.value)
    res = localize(state, inst)
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
#     # Happens if attr is a method
#     print("@@@@@", a)
#     return a

@dy
def simplify(state: State, funccall: ast.FuncCall):
    func = simplify(state, funccall.func)

    matched = func.match_params(funccall.args)
    # args = [(p, simplify(state, a)) for p, a in matched]
    args = matched
    if isinstance(func, objects.UserFunction):
        with state.use_scope({p.name:simplify(state, a) for p,a in args}):
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
@dy
def simplify(state: State, o: ast.Order):
    return compile_remote(state, o)

@dy
def simplify(state: State, n: types.NullType):
    return n


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
            v = localize(state, evaluate(state, v))
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
    # res = localize(state, inst)
    # return promise(state, inst)
    return inst


def localize(session, inst):
    if isinstance(inst, objects.Function):
        return inst

    res = session.db.query(inst.code, inst.subqueries)

    if isinstance(inst.type, types.ListType):
        assert all(len(e)==1 for e in res)
        return [e['value'] for e in res]

    return res

