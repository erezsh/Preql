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

from copy import copy
from typing import List, Optional, Any
import logging
import re

from .utils import benchmark
from .utils import safezip, dataclass, SafeDict, listgen
from .interp_common import assert_type, exclude_fields
from .exceptions import pql_TypeError, pql_ValueError, pql_NameNotFound, ReturnSignal, pql_AttributeError, PreqlError, pql_SyntaxError, pql_CompileError
from . import exceptions as exc
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from . import settings
RawSql = sql.RawSql
Sql = sql.Sql

from .interp_common import State, dy, new_value_instance
from .compiler import compile_to_inst, compile_type_def



@dy
def resolve(state: State, struct_def: ast.StructDef):
    members = {str(k):resolve(state, v) for k, v in struct_def.members}
    struct = types.StructType(struct_def.name, members)
    state.set_var(struct_def.name, struct)
    return struct

@dy
def resolve(state: State, table_def: ast.TableDef) -> types.TableType:
    t = types.TableType(table_def.name, SafeDict(), False, [['id']])
    if table_def.methods:
        methods = evaluate(state, table_def.methods)
        t.attrs.update({m.userfunc.name:m.userfunc for m in methods})

    state.set_var(t.name, t)   # TODO use an internal namespace

    t.columns['id'] = types.IdType(t)
    for c in table_def.columns:
        t.columns[c.name] = resolve(state, c)

    state.set_var(t.name, objects.new_table(t))
    return t

@dy
def resolve(state: State, col_def: ast.ColumnDef):
    col = resolve(state, col_def.type)

    query = col_def.query
    if isinstance(col, objects.TableInstance):
        col = col.type
        assert isinstance(col, types.TableType)
    if col.composed_of(types.TableType):
        return types.RelationalColumn(col, query)

    assert not query
    return types.DataColumn(col, col_def.default)

@dy
def resolve(state: State, type_: ast.Type) -> types.PqlType:
    t = state.get_var(type_.name)
    if type_.nullable:
        t = types.OptionalType(t)
    return t


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
def _set_value(state: State, name: ast.Name, value):
    state.set_var(name.name, value)

@dy
def _set_value(state: State, attr: ast.Attr, value):
    raise NotImplementedError("")

@dy
def _execute(state: State, var_def: ast.SetValue):
    res = evaluate(state, var_def.value)
    # res = apply_database_rw(state, res)
    _set_value(state, var_def.name, res)


@dy
def _copy_rows(state: State, target_name: ast.Name, source: objects.TableInstance):

    if source is objects.EmptyList: # Nothing to add
        return objects.null

    target = evaluate(state, target_name)

    params = dict(target.type.params())
    for p in params:
        if p not in source.type.columns:
            raise TypeError(None, f"Missing column {p} in table {source}")

    # assert len(params) == len(source.type.columns), (params, source)
    primary_keys, columns = target.type.flat_for_insert()

    source = exclude_fields(state, source, primary_keys)

    code = sql.Insert(target.type, columns, source.code)
    state.db.query(code, source.subqueries)
    return objects.null

@dy
def _execute(state: State, insert_rows: ast.InsertRows):
    if not isinstance(insert_rows.name, ast.Name):
        # TODO support Attr
        raise pql_SyntaxError(insert_rows.meta, "L-value must be table name")

    rval = evaluate(state, insert_rows.value)
    return _copy_rows(state, insert_rows.name, rval)

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
    return objects.null

@dy
def _execute(state: State, i: ast.If):
    cond = localize(state, evaluate(state, i.cond))
    if cond:
        execute(state, i.then)
    elif i.else_:
        execute(state, i.else_)

@dy
def _execute(state: State, f: ast.For):
    expr = localize(state, evaluate(state, f.iterable))
    for i in expr:
        with state.use_scope({f.var: objects.from_python(i)}):
            execute(state, f.do)

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
    e = evaluate(state, t.value)
    if isinstance(e, ast.Ast):
        raise exc.InsufficientAccessLevel()
    assert isinstance(e, Exception), e
    raise e

def execute(state, stmt):
    try:
        if isinstance(stmt, ast.Statement):
            return _execute(state, stmt) or objects.null
        return evaluate(state, stmt)
    except PreqlError as e:
        if e.meta is None:
            raise e.replace(meta=stmt.meta)
        raise #e.replace(meta=e.meta.replace(parent=stmt.meta))




# TODO Is simplify even helpful? Why not just compile everything?
#      evaluate() already compiles anyway, so what's the big deal?
#      an "optimization" for local tree folding can be added later,
#      and also prevent compiling lists (which simplify doesn't prevent)

@dy
def simplify(state: State, cb: ast.CodeBlock):
    return _execute(state, cb)

@dy
def simplify(state: State, n: ast.Name):
    # XXX what happens to caching if this is a global variable?
    return state.get_var(n.name)

@dy
def simplify(state: State, x):
    return x

# @dy
# def simplify(state: State, ls: list):
#     return [simplify(state, i) for i in ls]

# @dy
# def simplify(state: State, d: objects.ParamDict):
#     return d.replace(params={name: evaluate(state, v) for name, v in d.params.items()})

# @dy
# def simplify(state: State, node: ast.Ast):
#     # TODO implement automatically with prerequisites
#     # return _simplify_ast(state, node)
#     return node

# def _simplify_ast(state, node):
#     resolved = {k:simplify(state, v) for k, v in node
#                 if isinstance(v, types.PqlObject) or isinstance(v, list) and all(isinstance(i, types.PqlObject) for i in v)}
#     return node.replace(**resolved)

# @dy
# def simplify(state: State, cb: ast.CodeBlock):
#     # if len(cb.statements) == 1:
#     #     return simplify(state, cb.statements[0])
#     return _simplify_ast(state, cb)

# @dy
# def simplify(state: State, if_: ast.If):
#     if_ = _simplify_ast(state, if_)
#     if isinstance(if_.cond, objects.ValueInstance): # XXX a more general test?
#         if if_.cond.local_value:
#             return if_.then
#         else:
#             return if_.else_
#     return if_

# TODO isn't this needed somewhere??
# @dy
# def simplify(state: State, obj: ast.Or):
#     for expr in obj.args:
#         inst = evaluate(state, expr)
#         nz = test_nonzero(state, inst)
#         if nz:
#             return inst
#     return inst



@dy
def simplify(state: State, funccall: ast.FuncCall):
    # func = simplify(state, funccall.func)
    func = evaluate(state, funccall.func)

    if isinstance(func, types.Primitive):
        # Cast to primitive
        assert func is types.Int
        func = state.get_var('_cast_int')

    if not isinstance(func, objects.Function):
        meta = funccall.func.meta
        meta = meta.replace(parent=meta)
        raise pql_TypeError(meta, f"Error: Object of type '{func.type}' is not callable")

    return eval_func_call(state, func, funccall.args, funccall.meta)


def eval_func_call(state, func, args, meta=None):
    assert isinstance(func, objects.Function)

    matched_args = func.match_params(args)



    if isinstance(func, objects.MethodInstance):
        args = {'this': func.parent}
        # args.update(func.parent.all_attrs())
    else:
        args = {}

    args.update( {p.name:simplify(state, a) for p,a in matched_args} )


    # if isinstance(func, objects.UserFunction):
    if isinstance(func, objects.InternalFunction):
        # TODO ensure pure function?
        # TODO Ensure correct types
        return func.func(state, *args.values())
    else:
        # TODO make tests to ensure caching was successful
        if settings.cache:
            params = {name: ast.Parameter(meta, name, value.type) for name, value in args.items()}
            sig = (func.name,) + tuple(a.type for a in args.values())

            try:
                # with state.use_scope(args):
                with state.use_scope(params):
                    if sig in state._cache:
                        expr = state._cache[sig]
                    else:
                        logging.info(f"Compiling.. {func}")
                        expr = _call_expr(state.reduce_access(state.AccessLevels.COMPILE), func.expr)
                        logging.info("Compiled successfully")
                        state._cache[sig] = expr

                expr = ast.ResolveParameters(meta, expr, args)
            except exc.InsufficientAccessLevel:
                # Don't cache
                expr = func.expr
        else:
            expr = func.expr

        with state.use_scope(args):
            with benchmark.measure('call_expr'):
                res = _call_expr(state, expr)

            if isinstance(res, ast.ResolveParameters):  # XXX A bit of a hack
                raise exc.InsufficientAccessLevel()

            return res


def _call_expr(state, expr):
    try:
        return evaluate(state, expr)
    except ReturnSignal as r:
        return r.value

@dy
def resolve_parameters(state: State, x):
    return x

@dy
def resolve_parameters(state: State, res: ast.ResolveParameters):

    # XXX use a different mechanism??
    if isinstance(res.obj, objects.Instance):
        obj = res.obj
    else:
        with state.use_scope(res.values):
            obj = evaluate(state, res.obj)

    state.require_access(state.AccessLevels.WRITE_DB)

    if not isinstance(obj, objects.Instance):
        if isinstance(obj, objects.Function):
            return obj
        return res.replace(obj=obj)

    code = _resolve_sql_parameters(state, obj.code)

    return obj.replace(code=code)

@dy
def test_nonzero(state: State, table: objects.TableInstance):
    count = call_pql_func(state, "count", [table])
    return localize(state, evaluate(state, count))

@dy
def test_nonzero(state: State, inst: objects.ValueInstance):
    return bool(inst.local_value)

def _raw_sql_callback(state: State, var: str, instances):
    var = var.group()
    assert var[0] == '$'
    var_name = var[1:]
    obj = state.get_var(var_name)

    if isinstance(obj, types.TableType):
        # This branch isn't strictly necessary
        # It exists to create nicer SQL code output
        inst = objects.new_table(obj)
    else:
        inst = evaluate(state, obj)

    instances.append(inst)

    qb = sql.QueryBuilder(state.db.target, False)
    code = _resolve_sql_parameters(state, inst.code)
    return '%s' % code.compile(qb).text



@dy
def resolve_parameters(state: State, p: ast.Parameter):
    return state.get_var(p.name)


# TODO move this to SQL compilation??
@dy
def apply_database_rw(state: State, rps: ast.ResolveParametersString):
    # TODO if this is still here, it should be in evaluate, not db_rw
    state.catch_access(state.AccessLevels.EVALUATE)

    sql_code = localize(state, evaluate(state, rps.string))
    assert isinstance(sql_code, str)

    type_ = evaluate(state, rps.type)
    if isinstance(type_, objects.Instance):
        type_ = type_.type
    assert isinstance(type_, types.PqlType), type_

    instances = []
    expanded = re.sub(r"\$\w+", lambda m: _raw_sql_callback(state, m, instances), sql_code)
    code = sql.RawSql(type_, expanded)
    # code = sql.ResolveParameters(sql_code)

    # TODO validation!!
    if isinstance(type_, types.TableType):
        name = state.unique_name("subq_")

        # TODO this isn't in the tests!
        fields = [sql.Name(c, path) for path, c in type_.flatten_type()]

        subq = sql.Subquery(name, fields, code)

        inst = objects.new_table(type_, name, instances)
        inst.subqueries[name] = subq
        return inst

    return objects.Instance.make(code, type_, instances)


@dy
def apply_database_rw(state: State, o: ast.One):
    # TODO move these to the core/base module
    table = evaluate(state, ast.Slice(None, o.expr, ast.Range(None, None, ast.Const(None, types.Int, 2))))
    if isinstance(table, ast.Ast):
        return table
    if isinstance(table.type, types.NullType):
        return table

    assert isinstance(table.type, types.Collection), table
    rows = localize(state, table) # Must be 1 row
    if len(rows) == 0:
        if not o.nullable:
            raise pql_ValueError(o.meta, "'one' expected a single result, got an empty expression")
        return objects.null
    elif len(rows) > 1:
        raise pql_ValueError(o.meta, "'one' expected a single result, got more")

    row ,= rows
    rowtype = types.RowType(table.type)
    # XXX ValueInstance is the right object? Why not throw away 'code'? Reusing it is inefficient
    return objects.ValueInstance.make(table.code, rowtype, [table], row)

@dy
def apply_database_rw(state: State, d: ast.Delete):
    state.catch_access(state.AccessLevels.WRITE_DB)
    # TODO Optimize: Delete on condition, not id, when possible

    cond_table = ast.Selection(d.meta, d.table, d.conds)
    table = evaluate(state, cond_table)
    assert isinstance(table.type, types.TableType)

    rows = list(localize(state, table))
    if rows:
        if 'id' not in rows[0]:
            raise pql_ValueError(d.meta, "Delete error: Table does not contain id")

        ids = [row['id'] for row in rows]

        for code in sql.deletes_by_ids(table, ids):
            state.db.query(code, table.subqueries)

    return evaluate(state, d.table)

@dy
def apply_database_rw(state: State, u: ast.Update):
    state.catch_access(state.AccessLevels.WRITE_DB)

    # TODO Optimize: Update on condition, not id, when possible
    table = evaluate(state, u.table)
    assert isinstance(table.type, types.TableType)
    assert all(f.name for f in u.fields)

    # TODO verify table is concrete (i.e. lvalue, not a transitory expression)
    # try:
    #     state.get_var(table.type.name)
    # except pql_NameNotFound:
    #     meta = u.table.meta
    #     raise pql_TypeError(meta.replace(meta), "Update error: Got non-real table")

    update_scope = {n:c.replace(code=sql.Name(c.type, n)) for n, c in table.all_attrs().items()}
    with state.use_scope(update_scope):
        proj = {f.name:evaluate(state, f.value) for f in u.fields}

    rows = list(localize(state, table))
    if rows:
        if 'id' not in rows[0]:
            raise pql_ValueError(u.meta, "Update error: Table does not contain id")
        if not set(proj) < set(rows[0]):
            raise pql_ValueError(u.meta, "Update error: Not all keys exist in table")

        ids = [row['id'] for row in rows]

        for code in sql.updates_by_ids(table, proj, ids):
            state.db.query(code, table.subqueries)

    # TODO return by ids to maintain consistency, and skip a possibly long query
    return table


@dy
def apply_database_rw(state: State, new: ast.NewRows):
    state.catch_access(state.AccessLevels.WRITE_DB)

    obj = state.get_var(new.type)

    if isinstance(obj, objects.TableInstance):
        # XXX Is it always TableInstance? Just sometimes? What's the transition here?
        obj = obj.type
    assert_type(new.meta, obj, types.TableType, "'new' expected an object of type '%s', instead got '%s'")

    if len(new.args) > 1:
        raise NotImplementedError("Not yet implemented. Requires column-wise table concat (use join and enum)")

    arg ,= new.args

    # TODO postgres can do it better!
    field = arg.name
    table = evaluate(state, arg.value)
    rows = localize(state, table)

    cons = TableConstructor.make(obj)

    # TODO very inefficient, vectorize this
    ids = []
    for row in rows:
        matched = cons.match_params([objects.from_python(v) for v in row.values()])
        destructured_pairs = _destructure_param_match(state, new.meta, matched)
        ids += [_new_row(state, obj, destructured_pairs)]

    # XXX find a nicer way - requires a better typesystem, where id(t) < int
    # return ast.List_(new.meta, [new_value_instance(rowid, obj.columns['id'], force_type=True) for rowid in ids])
    return ast.List_(new.meta, types.ListType(types.Int), ids)


@listgen
def _destructure_param_match(state, meta, param_match):
    # TODO use cast rather than a ad-hoc hardwired destructure
    for k, v in param_match:
        v = localize(state, evaluate(state, v))
        if isinstance(k.type, types.StructType):
            names = [name for name, t in k.orig.col_type.flatten_type([k.name])]
            if not isinstance(v, list):
                raise pql_TypeError(meta, f"Parameter {k.name} received a bad value (expecting a struct or a list)")
            if len(v) != len(names):
                raise pql_TypeError(meta, f"Parameter {k.name} received a bad value (size of {len(names)})")
            yield from safezip(names, v)
        else:
            yield k.name, v

def _new_row(state, table, destructured_pairs):
    keys = [name for (name, _) in destructured_pairs]
    values = [sql.value(v) for (_,v) in destructured_pairs]
    # TODO use regular insert?
    q = sql.InsertConsts(sql.TableName(table, table.name), keys, values)
    state.db.query(q)
    rowid = state.db.query(sql.LastRowId())
    return new_value_instance(rowid, table.columns['id'], force_type=True)  # XXX find a nicer way


@dy
def apply_database_rw(state: State, new: ast.New):
    state.catch_access(state.AccessLevels.WRITE_DB)

    # XXX This function has side-effects.
    # Perhaps it belongs in resolve, rather than simplify?
    obj = state.get_var(new.type)

    # XXX Assimilate this special case
    if isinstance(obj, type) and issubclass(obj, PreqlError):
        def create_exception(state, msg):
            msg = localize(state, evaluate(state, msg))
            return obj(None, msg)
        f = objects.InternalFunction(obj.__name__, [objects.Param(None, 'message')], create_exception)
        res = evaluate(state, ast.FuncCall(new.meta, f, new.args))
        return res

    assert isinstance(obj, objects.TableInstance)  # XXX always the case?
    table = obj
    # TODO assert tabletype is a real table and not a query (not transient), otherwise new is meaningless
    assert_type(new.meta, table.type, types.TableType, "'new' expected an object of type '%s', instead got '%s'")

    cons = TableConstructor.make(table.type)
    matched = cons.match_params(new.args)

    destructured_pairs = _destructure_param_match(state, new.meta, matched)
    rowid = _new_row(state, table.type, destructured_pairs)
    return rowid
    # return new_value_instance(rowid, table.type.columns['id'], force_type=True)  # XXX find a nicer way
    # expr = ast.One(None, ast.Selection(None, table, [ast.Compare(None, '==', [ast.Name(None, 'id'), new_value_instance(rowid)])]), False)
    # return evaluate(state, expr)


@dataclass
class TableConstructor(objects.Function):
    "Serves as an ad-hoc constructor function for given table, to allow matching params"

    params: List[objects.Param]
    param_collector: Optional[objects.Param] = None
    name = 'new'

    @classmethod
    def make(cls, table):
        return cls([objects.Param(name.meta, name, p.col_type, p.default, orig=p) for name, p in table.params()])


def add_as_subquery(state: State, inst: objects.Instance):
    code_cls = sql.TableName if isinstance(inst.type, types.Collection) else sql.Name
    name = state.unique_name(inst)
    return inst.replace(code=code_cls(inst.code.type, name), subqueries=inst.subqueries.update({name: inst.code}))


@dy
def evaluate(state, obj: list):
    return [evaluate(state, item) for item in obj]

@dy
def evaluate(state, obj_):
    obj = simplify(state, obj_)
    assert obj, obj_

    if state.access_level < state.AccessLevels.COMPILE:
        return obj

    # obj = compile_to_inst(state.reduce_access(state.AccessLevels.COMPILE), obj)
    obj = compile_to_inst(state, obj)

    if state.access_level < state.AccessLevels.EVALUATE:
        return obj

    obj = resolve_parameters(state, obj)

    if state.access_level < state.AccessLevels.READ_DB:
        return obj

    # Apply read-write operations XXX still needs rethinking. For example, can't 'One' be lazy?
    obj = apply_database_rw(state, obj)

    assert not isinstance(obj, (ast.ResolveParameters, ast.ResolveParametersString))

    return obj


@dy
def apply_database_rw(state, x):
    return x

#
#    localize()
# -------------
#
# Return the local value of the expression. Only requires computation if the value is an instance.
#
from copy import copy
@dy
def __resolve_sql_parameters(ns, param: sql.Parameter):
    inst = ns.get_var(param.name)
    assert isinstance(inst, objects.Instance)
    assert inst.type == param.type
    ns = type(ns)(ns.ns[-1])
    return __resolve_sql_parameters(ns, inst.code)

@dy
def __resolve_sql_parameters(ns, l: list):
    return [__resolve_sql_parameters(ns, n) for n in l]

@dy
def __resolve_sql_parameters(ns, node):
    resolved = {k:__resolve_sql_parameters(ns, v) for k, v in node
                if isinstance(v, Sql) or isinstance(v, list) and all(isinstance(i, Sql) for i in v)}
    return node.replace(**resolved)

def _resolve_sql_parameters(state, node):
    # 1. Resolve parameters while compiling
    return sql.ResolveParameters(node, copy(state.ns))
    # 2. Resolve parameters before compiling. Eqv to (1) but slower
    # return __resolve_sql_parameters(state.ns, node)


@dy
def localize(state, inst: objects.Instance):
    state.require_access(state.AccessLevels.WRITE_DB)

    # code = _resolve_sql_parameters(state, inst.code)

    return state.db.query(inst.code, inst.subqueries, state=state)

@dy
def localize(state, inst: objects.ValueInstance):
    return inst.local_value

@dy
def localize(state, x):
    return x


