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
from .interp_common import assert_type
from .exceptions import pql_TypeError, pql_ValueError, pql_NameNotFound, ReturnSignal, pql_AttributeError, PreqlError
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
RawSql = sql.RawSql
Sql = sql.Sql

from .interp_common import State, dy, get_alias
from .compiler import compile_remote, compile_type_def, sql_repr, instanciate_table



@dy
def resolve(state: State, struct_def: ast.StructDef):
    members = {str(k):resolve(state, v) for k, v in struct_def.members}
    struct = types.StructType(struct_def.name, members)
    state.set_var(struct_def.name, struct)
    return struct

@dy
def resolve(state: State, table_def: ast.TableDef) -> types.TableType:
    t = types.TableType(table_def.name, {}, False)
    state.set_var(t.name, objects.InstancePlaceholder(t))

    t.add_column(types.DatumColumnType("id", types.IdType(t), primary_key=True, readonly=True))
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
    sql = compile_type_def(state, t)
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

@dy
def _execute(state: State, cb: ast.CodeBlock):
    for stmt in cb.statements:
        execute(state, stmt)

@dy
def _execute(state: State, i: ast.If):
    cond = localize(state, evaluate(state, i.cond))
    if cond:
        execute(state, i.then)
    elif i.else_:
        execute(state, i.else_)

@dy
def _execute(state: State, t: ast.Try):
    try:
        execute(state, t.try_)
    except PreqlError as e:
        exc_type = localize(state, evaluate(state, t.catch_expr))
        if isinstance(e, exc_type):
            execute(state, t.catch_block)
        else:
            raise

@dy
def _execute(state: State, r: ast.Return):
    value = evaluate(state, r.value)
    raise ReturnSignal(value)

@dy
def _execute(state: State, t: ast.Throw):
    raise evaluate(state, t.value)

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
    obj = state.get_var(n.name)
    # if isinstance(obj, types.TableType):
    #     return instanciate_table(state, obj, sql.TableName(obj, obj.name), [])
    return obj

@dy
def simplify(state: State, c: ast.Const):
    return c

@dy
def simplify(state: State, a: ast.Attr):
    obj = simplify(state, a.expr)
    return ast.Attr(a.meta, obj, a.name)

#     assert isinstance(obj, types.PqlType)   # Only tested on types so far
#     if a.name == 'name':
#         return ast.Const(a.meta, types.String, obj.name)
#     raise pql_AttributeError(a.meta, "Type '%s' has no attribute '%s'" % (obj, a.name))

@dy
def simplify(state: State, funccall: ast.FuncCall):
    # func = simplify(state, funccall.func)
    func = compile_remote(state, funccall.func)

    if isinstance(func, types.Primitive):
        # Cast to primitive
        assert func is types.Int
        func = state.get_var('_cast_int')

    if not isinstance(func, objects.Function):
        meta = funccall.func.meta
        meta = meta.remake(parent=meta)
        raise pql_TypeError(meta, f"Error: Object of type '{func.type.concrete_type()}' is not callable")

    args = func.match_params(funccall.args)
    if isinstance(func, objects.UserFunction):
        with state.use_scope({p.name:simplify(state, a) for p,a in args}):
            if isinstance(func.expr, ast.CodeBlock):
                # from .evaluate import execute   # XXX TODO fix this
                try:
                    return execute(state, func.expr)
                except ReturnSignal as r:
                    return r.value

            r = simplify(state, func.expr)
            return r
    else:
        return func.func(state, *[v for k,v in args])




@dy
def simplify(state: State, c: ast.Arith):
    return ast.Arith(c.meta, c.op, simplify_list(state, c.args))

@dy
def simplify(state: State, c: ast.CodeBlock):
    return ast.CodeBlock(c.meta, simplify_list(state, c.statements))

@dy
def simplify(state: State, x: ast.If):
    # TODO if cond can be simplified to a constant, just cull either then or else
    return x
@dy
def simplify(state: State, x: ast.Throw):
    return x

@dy
def simplify(state: State, c: objects.List_):
    return objects.List_(c.meta, simplify_list(state, c.elems))

@dy
def simplify(state: State, c: ast.Compare):
    return ast.Compare(c.meta, c.op, simplify_list(state, c.args))

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
def simplify(state: State, u: ast.Update):
    table = evaluate(state, u.table)
    assert isinstance(table, objects.TableInstance)
    assert all(f.name for f in u.fields)

    # TODO verify table is concrete (i.e. lvalue, not a transitory expression)
    # try:
    #     state.get_var(table.type.name)
    # except pql_NameNotFound:
    #     meta = u.table.meta
    #     raise pql_TypeError(meta.remake(meta), "Update error: Got non-real table")

    update_scope = {n:c.remake(code=sql.Name(c.type.concrete_type(), n)) for n, c in table.columns.items()}
    with state.use_scope(update_scope):
        proj = {f.name:compile_remote(state, f.value) for f in u.fields}
    sql_proj = {sql.Name(value.type, name): value.code for name, value in proj.items()}
    for row in localize(state, table):
        id_ = row['id']
        if not set(proj) < set(row):
            raise pql_ValueError("Update error: Not all keys exist in table")
        compare = sql.Compare(types.Bool, '=', [sql.Name(types.Int, 'id'), sql.Primitive(types.Int, str(id_))])
        code = sql.Update(types.null, sql.TableName(table.type, table.type.name), sql_proj, [compare])
        state.db.query(code, table.subqueries)

    # TODO return by ids to maintain consistency, and skip a possibly long query
    return table


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
def simplify(state: State, inst: objects.Instance):
    return inst
@dy
def simplify(state: State, inst: objects.ValueInstance):
    return inst
@dy
def simplify(state: State, inst: objects.TableInstance):
    return inst

@dy
def simplify(state: State, new: ast.New):
    # XXX This function has side-effects.
    # Perhaps it belongs in resolve, rather than simplify?
    obj = state.get_var(new.type)

    if isinstance(obj, type) and issubclass(obj, PreqlError):
        def create_exception(state, msg):
            msg = localize(state, compile_remote(state, msg))
            return obj(None, msg)
        f = objects.InternalFunction(obj.__name__, [objects.Param(None, 'message')], create_exception)
        res = simplify(state, ast.FuncCall(new.meta, f, new.args))
        return res

    obj = obj.concrete_type()
    assert_type(new.meta, obj, types.TableType, "'new' expected an object of type '%s', instead got '%s'")
    table = obj

    cons = TableConstructor(list(table.params()))
    matched = cons.match_params(new.args)

    destructured_pairs = []
    for k, v in matched:
        if isinstance(k.type, types.StructType):
            v = localize(state, evaluate(state, v))
            for k2, v2 in safezip(k.flatten(), v):
                destructured_pairs.append((k2, v2))
        else:
            v = localize(state, evaluate(state, v))
            destructured_pairs.append((k, v))

    keys = [k.name for (k,_) in destructured_pairs]
    values = [sql_repr(v) for (_,v) in destructured_pairs]
    # sql = RawSql(f"INSERT INTO {table.name} ($keys_str) VALUES ($values_str)")
    q = sql.InsertConsts(types.null, sql.TableName(table, table.name), keys, values)

    state.db.query(q)
    rowid = state.db.query(sql.LastRowId())
    return ast.Const(None, types.Int, rowid)   # Todo Row reference / query



def add_as_subquery(state: State, inst: objects.Instance):
    name = get_alias(state, inst)
    if isinstance(inst, objects.TableInstance):
        new_inst = objects.TableInstance(sql.TableName(inst.code.type, name), inst.type, inst.subqueries.update({name: inst.code}), inst.columns)
    else:
        new_inst = objects.Instance(sql.Name(inst.code.type, name), inst.type, inst.subqueries.update({name: inst.code}))
    return new_inst



def evaluate(state, obj):
    obj = simplify(state, obj)
    if isinstance(obj, (objects.Function, PreqlError, type)):   # TODO base class on uncompilable?
        return obj

    inst = compile_remote(state, obj)
    # res = localize(state, inst)
    # return promise(state, inst)
    return inst


def localize(session, inst):
    assert inst
    if isinstance(inst, objects.Function) or isinstance(inst, (types.PqlType, type)):
        return inst

    elif isinstance(inst, objects.ValueInstance):
        return inst.local_value

    res = session.db.query(inst.code, inst.subqueries)

    # if isinstance(inst.type, types.ListType):
    #     # assert False
    #     assert all(len(e)==1 for e in res)
    #     return [e['value'] for e in res]

    return res

