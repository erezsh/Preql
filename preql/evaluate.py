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

from .utils import safezip, dataclass, SafeDict, listgen
from .interp_common import assert_type
from .exceptions import pql_TypeError, pql_ValueError, pql_NameNotFound, ReturnSignal, pql_AttributeError, PreqlError, pql_SyntaxError
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
RawSql = sql.RawSql
Sql = sql.Sql

from .interp_common import State, dy, get_alias, sql_repr, make_value_instance
from .compiler import compile_remote, compile_type_def, instanciate_table, call_pql_func, exclude_fields



@dy
def resolve(state: State, struct_def: ast.StructDef):
    members = {str(k):resolve(state, v) for k, v in struct_def.members}
    struct = types.StructType(struct_def.name, members)
    state.set_var(struct_def.name, struct)
    return struct

@dy
def resolve(state: State, table_def: ast.TableDef) -> types.TableType:
    t = types.TableType(table_def.name, SafeDict(), False, [['id']])
    state.set_var(t.name, t)   # TODO use an internal namespace

    t.columns['id'] = types.IdType(t)
    for c in table_def.columns:
        t.columns[c.name] = resolve(state, c)

    inst = instanciate_table(state, t, sql.TableName(t, t.name), [])
    state.set_var(t.name, inst)
    return t

@dy
def resolve(state: State, col_def: ast.ColumnDef):
    col = resolve(state, col_def.type)

    query = col_def.query
    if isinstance(col, objects.TableInstance):
        return types.RelationalColumn(col.type, query)

    assert not query
    return types.DatumColumn(col, col_def.default)

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
def _set_value(state: State, name: ast.Name, value):
    state.set_var(name.name, value)

@dy
def _set_value(state: State, attr: ast.Attr, value):
    raise NotImplementedError("")

@dy
def _execute(state: State, var_def: ast.SetValue):
    res = simplify(state, var_def.value)
    res = resolve_effects(state, res)
    _set_value(state, var_def.name, res)



@dy
def _copy_rows(state: State, target_name: ast.Name, source: objects.TableInstance):

    if source is objects.EmptyList: # Nothing to add
        return objects.null

    target = simplify(state, target_name)

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

    rval = evaluate(state, simplify(state, insert_rows.value))
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
    raise evaluate(state, t.value)

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
def simplify(state: State, n: ast.Name):
    # Don't recurse simplify, to allow granular dereferences
    # The assumption is that the stored variable is already simplified
    obj = state.get_var(n.name)
    return obj



# XXX This is a big one!
@dy
def simplify(state: State, funccall: ast.FuncCall):
    # func = simplify(state, funccall.func)
    func = simplify(state, funccall.func)

    if isinstance(func, types.Primitive):
        # Cast to primitive
        assert func is types.Int
        func = state.get_var('_cast_int')

    if not isinstance(func, objects.Function):
        meta = funccall.func.meta
        meta = meta.replace(parent=meta)
        raise pql_TypeError(meta, f"Error: Object of type '{func.type}' is not callable")

    matched_args = func.match_params(funccall.args)
    args = {p.name:evaluate(state, a) for p,a in matched_args}
    try:
        sig = (funccall.func,) + tuple(a.type for a in args.values())
    except:
        raise

    if isinstance(func, objects.UserFunction):
        with state.use_scope(args):
            # if sig in _cache:
            #     expr = _cache[sig]
            # else:
            #     expr = evaluate(state.reduce_access(state.AccessLevels.COMPILE), func.expr)
            #     _cache[sig] = expr

            expr = evaluate(state.reduce_access(state.AccessLevels.COMPILE), func.expr)

            if isinstance(func.expr, ast.CodeBlock):
                try:
                    return execute(state, expr)
                except ReturnSignal as r:
                    return r.value
            else:
                r = evaluate(state, expr)
                return r
    else:
        # TODO ensure pure function
        return func.func(state, *args.values())

_cache = {}

@dy
def test_nonzero(state: State, table: objects.TableInstance):
    count = call_pql_func(state, "count", [table])
    return localize(state, evaluate(state, count))

@dy
def test_nonzero(state: State, inst: objects.ValueInstance):
    return bool(inst.local_value)

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
def resolve_effects(state: State, o: ast.One):
    # TODO move these to the core/base module
    fname = '_only_one_or_none' if o.nullable else '_only_one'
    f = state.get_var(fname)
    table = simplify(state, ast.FuncCall(o.meta, f, [o.expr]))  # TODO Use call_pql_func
    if isinstance(table.type, types.NullType):
        return table

    assert isinstance(table, objects.TableInstance), table
    row ,= localize(state, table) # Must be 1 row
    rowtype = types.RowType(table.type)
    # XXX ValueInstance is the right object? Why not throw away 'code'? Reusing it is inefficient
    return objects.ValueInstance.make(table.code, rowtype, [table], row)

@dy
def resolve_effects(state: State, d: ast.Delete):
    assert state.access_level >= state.AccessLevels.WRITE_DB
    # TODO Optimize: Delete on condition, not id, when possible

    cond_table = ast.Selection(d.meta, d.table, d.conds)
    table = evaluate(state, cond_table)
    assert isinstance(table, objects.TableInstance)

    for row in localize(state, table):
        if 'id' not in row:
            raise pql_ValueError(d.meta, "Delete error: Table does not contain id")
        id_ = row['id']

        compare = sql.Compare('=', [sql.Name(types.Int, 'id'), sql.Primitive(types.Int, str(id_))])
        code = sql.Delete(sql.TableName(table.type, table.type.name), [compare])
        state.db.query(code, table.subqueries)

    return evaluate(state, d.table)

@dy
def resolve_effects(state: State, u: ast.Update):
    assert state.access_level >= state.AccessLevels.WRITE_DB
    # TODO Optimize: Update on condition, not id, when possible
    table = evaluate(state, u.table)
    assert isinstance(table, objects.TableInstance)
    assert all(f.name for f in u.fields)

    # TODO verify table is concrete (i.e. lvalue, not a transitory expression)
    # try:
    #     state.get_var(table.type.name)
    # except pql_NameNotFound:
    #     meta = u.table.meta
    #     raise pql_TypeError(meta.replace(meta), "Update error: Got non-real table")

    update_scope = {n:c.replace(code=sql.Name(c.type, n)) for n, c in table.columns.items()}
    with state.use_scope(update_scope):
        proj = {f.name:evaluate(state, f.value) for f in u.fields}
    sql_proj = {sql.Name(value.type, name): value.code for name, value in proj.items()}
    for row in localize(state, table):
        if 'id' not in row:
            raise pql_ValueError(u.meta, "Update error: Table does not contain id")
        id_ = row['id']
        if not set(proj) < set(row):
            raise pql_ValueError(u.meta, "Update error: Not all keys exist in table")
        compare = sql.Compare('=', [sql.Name(types.Int, 'id'), sql.Primitive(types.Int, str(id_))])
        code = sql.Update(sql.TableName(table.type, table.type.name), sql_proj, [compare])
        state.db.query(code, table.subqueries)

    # TODO return by ids to maintain consistency, and skip a possibly long query
    return table


@dy
def simplify(state: State, x):
    return x

@dy
def simplify(state: State, ls: list):
    return [simplify(state, i) for i in ls]

@dy
def simplify(state: State, node: ast.Ast):
    # TODO implement automatically with prerequisites
    # return _simplify_ast(state, node)
    return node

@dy
def compile_remote(state: State, node: ast.Ast):
    return node
@dy
def compile_remote(state: State, x):
    return x


def _simplify_ast(state, node):
    resolved = {k:simplify(state, v) for k, v in node
                if isinstance(v, types.PqlObject) or isinstance(v, list) and all(isinstance(i, types.PqlObject) for i in v)}
    return node.replace(**resolved)

# @dy
# def compile_remote(state: State, cb: ast.CodeBlock):
#     return cb.replace(statements=compile_remote(state, cb.statements))

# @dy
# def compile_remote(state: State, i: ast.If):
#     return i.replace(cond=compile_remote(state, i.cond))


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


@dy
def resolve_effects(state: State, new: ast.NewRows):
    assert state.access_level >= state.AccessLevels.WRITE_DB
    obj = state.get_var(new.type)

    if isinstance(obj, objects.TableInstance):
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
    # return ast.List_(new.meta, [make_value_instance(rowid, obj.columns['id'], force_type=True) for rowid in ids])
    return ast.List_(new.meta, ids)


@listgen
def _destructure_param_match(state, meta, param_match):
    # TODO use cast rather than a ad-hoc hardwired destructure
    for k, v in param_match:
        v = localize(state, evaluate(state, v))
        if isinstance(k.type.actual_type(), types.StructType):
            names = [name for name, t in k.orig.flatten_type([k.name])]
            if not isinstance(v, list):
                raise pql_TypeError(meta, f"Parameter {k.name} received a bad value (expecting a struct or a list)")
            if len(v) != len(names):
                raise pql_TypeError(meta, f"Parameter {k.name} received a bad value (size of {len(names)})")
            yield from safezip(names, v)
        else:
            yield k.name, v

def _new_row(state, table, destructured_pairs):
    keys = [name for (name, _) in destructured_pairs]
    values = [sql_repr(v) for (_,v) in destructured_pairs]
    # TODO use regular insert?
    q = sql.InsertConsts(sql.TableName(table, table.name), keys, values)
    state.db.query(q)
    rowid = state.db.query(sql.LastRowId())
    # return rowid
    return make_value_instance(rowid, table.columns['id'], force_type=True)  # XXX find a nicer way


@dy
def resolve_effects(state: State, new: ast.New):
    assert state.access_level >= state.AccessLevels.WRITE_DB

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
    # return make_value_instance(rowid, table.type.columns['id'], force_type=True)  # XXX find a nicer way
    # expr = ast.One(None, ast.Selection(None, table, [ast.Compare(None, '==', [ast.Name(None, 'id'), make_value_instance(rowid)])]), False)
    # return evaluate(state, expr)


@dataclass
class TableConstructor(objects.Function):
    "Serves as an ad-hoc constructor function for given table, to allow matching params"

    params: List[objects.Param]
    param_collector: Optional[objects.Param] = None
    name = 'new'

    @classmethod
    def make(cls, table):
        return cls([objects.Param(name.meta, name, p, p.default, orig=p) for name, p in table.params()])

    # def match_params(self, args):
    #     return [(p.orig, v) for p, v in super().match_params(args)]


def add_as_subquery(state: State, inst: objects.Instance):
    code_cls = sql.TableName if isinstance(inst, objects.TableInstance) else sql.Name
    name = get_alias(state, inst)
    return inst.replace(code=code_cls(inst.code.type, name), subqueries=inst.subqueries.update({name: inst.code}))

@dy
def simplify(state: State, d: objects.ParamDict):
    return d.replace(params={name: evaluate(state, v) for name, v in d.params.items()})

@dy
def evaluate(state, obj):
    obj = simplify(state, obj)
    assert obj, obj

    if state.access_level < state.AccessLevels.COMPILE:
        return obj
    obj = compile_remote(state.reduce_access(state.AccessLevels.COMPILE), obj)

    if state.access_level < state.AccessLevels.EVALUATE:
        return obj
    obj = resolve_effects(state, obj)

    return obj


@dy
def resolve_effects(state, x):
    return x

#
#    localize()
# -------------
#
# Return the local value of the expression. Only requires computation if the value is an instance.
#
@dy
def localize(session, inst: objects.Instance):
    return session.db.query(inst.code, inst.subqueries)

@dy
def localize(state, inst: objects.ValueInstance):
    return inst.local_value

@dy
def localize(state, x):
    return x


