from typing import List, Optional
import logging
from pathlib import Path

from preql.utils import safezip, dataclass, SafeDict, listgen
from preql import settings

from .interp_common import assert_type, exclude_fields, call_builtin_func, is_global_scope, cast_to_python_string, cast_to_python_int
from .exceptions import InsufficientAccessLevel, ReturnSignal, Signal
from . import exceptions as exc
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .parser import Str
from .interp_common import State, dsp, pyvalue_inst, cast_to_python
from .compiler import compile_to_inst, cast_to_instance
from .pql_types import T, Type, Object, Id
from .types_impl import table_params, table_flat_for_insert, flatten_type, pql_repr, kernel_type

MODULES_PATH = Path(__file__).parent.parent / 'modules'


@dsp
def resolve(state: State, struct_def: ast.StructDef):
    members = {str(k):resolve(state, v) for k, v in struct_def.members}
    struct = T.struct(members)
    state.set_var(struct_def.name, struct)
    return struct

@dsp
def resolve(state: State, table_def: ast.TableDef):
    name = table_def.name
    if is_global_scope(state):
        name = state.db.qualified_name(name)
        temporary = False
    else:
        name = '__local_' + state.unique_name(name)
        temporary = True

    t = T.table({}, name=Id(name), temporary=temporary)

    with state.use_scope({table_def.name: t}):  # For self-reference
        elems = {c.name: resolve(state, c) for c in table_def.columns}
        t = t(elems)

    if table_def.methods:
        methods = evaluate(state, table_def.methods)
        t.methods.update({m.userfunc.name:m.userfunc for m in methods})

    return t

@dsp
def resolve(state: State, col_def: ast.ColumnDef):
    coltype = resolve(state, col_def.type)

    query = col_def.query
    assert not query

    if isinstance(coltype, objects.SelectedColumnInstance):
        table = coltype.parent.type
        if 'name' not in table.options:
            # XXX better test for persistence
            raise Signal.make(T.TypeError, col_def.type, "Tables provided as relations must be persistent.")

        x = T.t_relation[coltype.type](rel={'table': table, 'column': coltype.name, 'key': False})
        return x.replace(_nullable=coltype.type._nullable)  # inherit is_nullable (TODO: use sumtypes?)

    elif coltype <= T.table:
        if 'name' not in coltype.options:
            # XXX better test for persistence
            raise Signal.make(T.TypeError, col_def.type, "Tables provided as relations must be persistent.")

        x = T.t_relation[T.t_id.as_nullable()](rel={'table': coltype, 'column': 'id', 'key': True})
        return x.replace(_nullable=coltype._nullable)     # inherit is_nullable (TODO: use sumtypes?)

    return coltype(default=col_def.default)

@dsp
def resolve(state: State, type_: ast.Type):
    t = evaluate(state, type_.type_obj)
    if isinstance(t, objects.TableInstance):
        t = t.type

    if not isinstance(t, (Type, objects.SelectedColumnInstance)):
        raise Signal.make(T.TypeError, type_, f"Expected type in column definition. Instead got '{t}'")

    if type_.nullable:
        t = t.as_nullable()

    return t


@dsp
def _execute(state: State, struct_def: ast.StructDef):
    resolve(state, struct_def)


def db_query(state: State, sql_code, subqueries=None):
    try:
        return state.db.query(sql_code, subqueries)
    except exc.DatabaseQueryError as e:
        raise Signal.make(T.DbQueryError, None, e.args[0]) from e

def drop_table(state, table_type):
    name ,= table_type.options['name'].parts
    code = sql.compile_drop_table(state, name)
    return state.db.query(code, {})


@dsp
def _execute(state: State, table_def: ast.TableDefFromExpr):
    expr = cast_to_instance(state, table_def.expr)
    name = table_def.name
    if is_global_scope(state):
        temporary = False
    else:
        name = '__local_' + state.unique_name(name)
        temporary = True
    t = new_table_from_expr(state, name, expr, table_def.const, temporary)
    state.set_var(table_def.name, t)

@dsp
def _execute(state: State, table_def: ast.TableDef):
    if table_def.columns and isinstance(table_def.columns[-1], ast.Ellipsis):
        ellipsis = table_def.columns.pop()
    else:
        ellipsis = None

    if any(isinstance(c, ast.Ellipsis) for c in table_def.columns):
        # XXX why must it? just ensure it appears once
        raise Signal.make(T.SyntaxError, table_def, "Ellipsis must appear at the end")

    # Create type and a corresponding table in the database
    t = resolve(state, table_def)
    db_name = t.options['name']

    exists = state.db.table_exists(db_name.repr_name)
    if exists:
        assert not t.options['temporary']
        cur_type = state.db.import_table_type(db_name.repr_name, None if ellipsis else set(t.elems) | {'id'})

        if ellipsis:
            elems_to_add = {Str(n, ellipsis.text_ref): v for n, v in cur_type.elems.items() if n not in t.elems}
            # TODO what is primary key isn't included?
            t = t({**t.elems, **elems_to_add}, **cur_type.options)

        # Auto-add id only if it exists already and not defined by user
        if 'id' in cur_type.elems: #and 'id' not in t.elems:
            # t = t(dict(id=T.t_id, **t.elems), pk=[['id']])
            assert cur_type.elems['id'] <= T.primitive, cur_type.elems['id']
            t.elems['id'] = T.t_id

        for e_name, e1_type in t.elems.items():

            if e_name not in cur_type.elems:
                raise Signal.make(T.TypeError, table_def, f"Column '{e_name}' defined, but doesn't exist in database.")

            # e2_type = cur_type.elems[e_name]
            # XXX use can_cast() instead of hardcoding it
            # if not (e1_type <= e2_type or (e1_type <= T.t_id and e2_type <= T.int)):
            #     raise Signal.make(T.TypeError, table_def, f"Cannot cast column '{e_name}' from type '{e2_type}' to '{e1_type}'")

        inst = objects.new_table(t, db_name, select_fields=True)
    else:
        # Auto-add id by default
        elems = dict(t.elems)
        if 'id' not in elems:
            elems = {'id': T.t_id, **elems}
        t = t(elems, pk=[['id']])
        inst = objects.new_table(t, db_name)

    state.set_var(table_def.name, inst)

    if not exists:
        sql_code = sql.compile_type_def(state, db_name.repr_name, t)
        db_query(state, sql_code)

@dsp
def _set_value(state: State, name: ast.Name, value):
    state.set_var(name.name, value)

@dsp
def _set_value(state: State, attr: ast.Attr, value):
    raise Signal.make(T.NotImplementedError, attr, f"Cannot set attribute for {attr.expr.repr()}")

@dsp
def _execute(state: State, var_def: ast.SetValue):
    res = evaluate(state, var_def.value)
    # res = apply_database_rw(state, res)
    _set_value(state, var_def.name, res)
    return res


def _copy_rows(state: State, target_name: ast.Name, source: objects.TableInstance):

    if source is objects.EmptyList: # Nothing to add
        return objects.null

    target = evaluate(state, target_name)

    params = dict(table_params(target.type))
    for p in params:
        if p not in source.type.elems:
            raise Signal.make(T.TypeError, source, f"Missing column '{p}' in {source.type}")

    primary_keys, columns = table_flat_for_insert(target.type)

    source = exclude_fields(state, source, set(primary_keys) & set(source.type.elems))

    code = sql.Insert(target.type.options['name'], columns, source.code)
    db_query(state, code, source.subqueries)
    return objects.null

@dsp
def _execute(state: State, insert_rows: ast.InsertRows):
    if not isinstance(insert_rows.name, ast.Name):
        # TODO support Attr
        raise Signal.make(T.SyntaxError, insert_rows, "L-value must be table name")

    rval = evaluate(state, insert_rows.value)

    assert_type(rval.type, T.table, insert_rows, '+=')

    return _copy_rows(state, insert_rows.name, rval)

@dsp
def _execute(state: State, func_def: ast.FuncDef):
    func = func_def.userfunc
    assert isinstance(func, objects.UserFunction)

    new_params = []
    for p in func.params:
        if p.type:
            t = evaluate(state, p.type)
            p = p.replace(type=t)
        new_params.append(p)

    state.set_var(func.name, func.replace(params=new_params))

@dsp
def _execute(state: State, p: ast.Print):
    # TODO Can be done better. Maybe cast to ReprText?
    insts = evaluate(state, p.value)
    assert isinstance(insts, list)

    for inst in insts:
        # inst = evaluate(state, p.value)
        if inst.type <= T.string:
            repr_ = cast_to_python_string(state, inst)
        else:
            repr_ = inst.repr()

        state.display.print(repr_, end=" ")
    state.display.print("")

@dsp
def _execute(state: State, p: ast.Assert):
    res = cast_to_python(state, p.cond)
    if not res:
        # TODO pretty print values
        if isinstance(p.cond, ast.Compare):
            s = (' %s '%p.cond.op).join(str(evaluate(state, a).repr()) for a in p.cond.args)
        else:
            s = p.cond.repr()
        raise Signal.make(T.AssertError, p.cond, f"Assertion failed: {s}")

@dsp
def _execute(state: State, cb: ast.CodeBlock):
    for stmt in cb.statements:
        execute(state, stmt)
    return objects.null


@dsp
def _execute(state: State, i: ast.If):
    cond = cast_to_python(state, i.cond)

    if cond:
        execute(state, i.then)
    elif i.else_:
        execute(state, i.else_)

@dsp
def _execute(state: State, w: ast.While):
    while cast_to_python(state, w.cond):
        execute(state, w.do)

@dsp
def _execute(state: State, f: ast.For):
    expr = cast_to_python(state, f.iterable)
    for i in expr:
        with state.use_scope({f.var: objects.from_python(i)}):
            execute(state, f.do)

@dsp
def _execute(state: State, t: ast.Try):
    try:
        execute(state, t.try_)
    except Signal as e:
        catch_type = localize(state, evaluate(state, t.catch_expr))
        if not isinstance(catch_type, Type):
            raise Signal.make(T.TypeError, t.catch_expr, f"Catch expected type, got {t.catch_expr.type}")
        if e.type <= catch_type:
            scope = {t.catch_name: e} if t.catch_name else {}
            with state.use_scope(scope):
                execute(state, t.catch_block)
        else:
            raise


def find_module(module_name):
    paths = [MODULES_PATH, Path.cwd()]
    for path in paths:
        module_path =  (path / module_name).with_suffix(".pql")
        if module_path.exists():
            return module_path

    raise Signal.make(T.ImportError, r, "Cannot find module")


def import_module(state, r):
    module_path = find_module(r.module_path)

    assert state is state.interp.state
    i = state.interp.clone(use_core=r.use_core)

    state.stacktrace.append(r.text_ref)
    try:
        i.include(module_path)
    finally:
        assert state.stacktrace[-1] is r.text_ref
        state.stacktrace.pop()

    # Inherit module db (in case it called connect())
    state.db = i.state.db

    ns = i.state.ns
    assert len(ns) == 1
    return objects.Module(r.module_path, ns._ns[0])


@dsp
def _execute(state: State, r: ast.Import):
    module = import_module(state, r)
    state.set_var(r.as_name or r.module_path, module)
    return module

@dsp
def _execute(state: State, r: ast.Return):
    value = evaluate(state, r.value)
    raise ReturnSignal(value)

@dsp
def _execute(state: State, t: ast.Throw):
    e = evaluate(state, t.value)
    if isinstance(e, ast.Ast):
        raise exc.InsufficientAccessLevel()
    assert isinstance(e, Exception), e
    raise e

def execute(state, stmt):
    if isinstance(stmt, ast.Statement):
        return _execute(state, stmt) or objects.null
    return evaluate(state, stmt)




# Simplify performs local operations before any db-specific compilation occurs
# Technically not super useful at the moment, but makes conceptual sense.

@dsp
def simplify(state: State, cb: ast.CodeBlock):
    # if len(cb.statements) == 1:
    #     s ,= cb.statements
    #     return simplify(state, s)
    try:
        return _execute(state, cb)
    except ReturnSignal as r:
        # XXX is this correct?
        return r.value
    except Signal as e:
        # Failed to run it, so try to cast as instance
        # XXX order should be other way around!
        if e.type <= T.CastError:
            return compile_to_inst(state, cb)
        raise
    except InsufficientAccessLevel:
        return cb

@dsp
def simplify(state: State, n: ast.Name):
    # XXX what happens to caching if this is a global variable?
    return state.get_var(n.name)

@dsp
def simplify(state: State, x):
    return x

# @dsp
# def simplify(state: State, ls: list):
#     return [simplify(state, i) for i in ls]

# @dsp
# def simplify(state: State, d: objects.ParamDict):
#     return d.replace(params={name: evaluate(state, v) for name, v in d.params.items()})

# @dsp
# def simplify(state: State, node: ast.Ast):
#     # return _simplify_ast(state, node)
#     return node

# def _simplify_ast(state, node):
#     resolved = {k:simplify(state, v) for k, v in node
#                 if isinstance(v, types.PqlObject) or isinstance(v, list) and all(isinstance(i, types.PqlObject) for i in v)}
#     return node.replace(**resolved)

# TODO isn't this needed somewhere??
# @dsp
# def simplify(state: State, if_: ast.If):
#     if_ = _simplify_ast(state, if_)
#     if isinstance(if_.cond, objects.ValueInstance): # XXX a more general test?
#         if if_.cond.local_value:
#             return if_.then
#         else:
#             return if_.else_
#     return if_


# TODO Optimize these, right now failure to evaluate will lose all work
@dsp
def simplify(state: State, obj: ast.Or):
    a, b = evaluate(state, obj.args)
    ta = kernel_type(a.type)
    tb = kernel_type(b.type)
    if ta != tb:
        raise Signal.make(T.TypeError, obj, f"'or' operator requires both arguments to be of the same type, but got '{ta}' and '{tb}'.")
    try:
        if test_nonzero(state, a):
            return a
    except InsufficientAccessLevel:
        return obj
    return b


@dsp
def simplify(state: State, obj: ast.And):
    a, b = evaluate(state, obj.args)
    ta = kernel_type(a.type)
    tb = kernel_type(b.type)
    if ta != tb:
        raise Signal.make(T.TypeError, obj, f"'and' operator requires both arguments to be of the same type, but got '{ta}' and '{tb}'.")
    try:
        if not test_nonzero(state, a):
            return a
    except InsufficientAccessLevel:
        return obj
    return b


@dsp
def simplify(state: State, obj: ast.Not):
    inst = evaluate(state, obj.expr)
    try:
        nz = test_nonzero(state, inst)
    except InsufficientAccessLevel:
        return obj
    return objects.pyvalue_inst(not nz)



@dsp
def simplify(state: State, funccall: ast.FuncCall):
    func = evaluate(state, funccall.func)

    if isinstance(func, objects.UnknownInstance):
        # evaluate(state, [a.value for a in funccall.args])
        raise Signal.make(T.TypeError, funccall.func, f"Error: Object of type '{func.type}' is not callable")

    args = funccall.args
    if isinstance(func, Type):
        # Cast to type
        args = args + [func]
        func = state.get_var('cast')

    if not isinstance(func, objects.Function):
        raise Signal.make(T.TypeError, funccall.func, f"Error: Object of type '{func.type}' is not callable")

    state.stacktrace.append(funccall.text_ref)
    try:
        res = eval_func_call(state, func, args)
    finally:
        assert state.stacktrace[-1] is funccall.text_ref
        state.stacktrace.pop()

    assert isinstance(res, Object), (type(res), res)
    return res


def eval_func_call(state, func, args):
    assert isinstance(func, objects.Function)

    matched_args = func.match_params(state, args)

    if isinstance(func, objects.MethodInstance):
        args = {'this': func.parent}
        # args.update(func.parent.all_attrs())
    else:
        args = {}

    # XXX simplify destroys text_ref, so it harms error messages.
    # TODO Can I get rid of it, or make it preserve the text_ref somehow?
    # Don't I need an instance to ensure I have type?

    for i, (p, a) in enumerate(matched_args):
        if not p.name.startswith('$'):      # $param means don't evaluate expression, leave it to the function
            a = evaluate(state, a)
        # TODO cast?
        if p.type and not a.type <= p.type:
            raise Signal.make(T.TypeError, func, f"Argument #{i} of '{func.name}' is of type '{a.type}', expected '{p.type}'")
        args[p.name] = a


    if isinstance(func, objects.InternalFunction):
        # TODO ensure pure function?
        # TODO Ensure correct types
        args = list(args.values())
        return func.func(state, *args)

    # TODO make tests to ensure caching was successful
    expr = func.expr
    if settings.cache:
        params = {name: ast.Parameter(name, value.type) for name, value in args.items()}
        sig = (func.name,) + tuple(a.type for a in args.values())

        try:
            with state.use_scope(params):
                if sig in state._cache:
                    compiled_expr = state._cache[sig]
                else:
                    logging.info(f"Compiling.. {func}")
                    compiled_expr = _call_expr(state.reduce_access(state.AccessLevels.COMPILE), func.expr)
                    logging.info("Compiled successfully")
                    if isinstance(compiled_expr, objects.Instance):
                        # XXX a little ugly
                        qb = sql.QueryBuilder(state.db.target, True)
                        x = compiled_expr.code.compile(qb)
                        x = x.optimize()
                        compiled_expr = compiled_expr.replace(code=x)
                    state._cache[sig] = compiled_expr

            expr = ast.ResolveParameters(compiled_expr, args)

        except exc.InsufficientAccessLevel:
            # Don't cache
            pass

    with state.use_scope(args):
        res = _call_expr(state, expr)

    if isinstance(res, ast.ResolveParameters):  # XXX A bit of a hack
        raise exc.InsufficientAccessLevel()

    return res


def _call_expr(state, expr):
    try:
        return evaluate(state, expr)
    except ReturnSignal as r:
        return r.value

# TODO fix these once we have proper types
@dsp
def test_nonzero(state: State, table: objects.TableInstance):
    count = call_builtin_func(state, "count", [table])
    return bool(cast_to_python_int(state, count))

@dsp
def test_nonzero(state: State, inst: objects.Instance):
    return bool(cast_to_python(state, inst))

@dsp
def test_nonzero(state: State, inst: Type):
    return True








@dsp
def apply_database_rw(state: State, o: ast.One):
    # TODO move these to the core/base module
    obj = evaluate(state, o.expr)
    if obj.type <= T.struct:
        if len(obj.attrs) != 1:
            raise Signal.make(T.ValueError, o, f"'one' expected a struct with a single attribute, got {len(obj.attrs)}")
        x ,= obj.attrs.values()
        return x

    slice_ast = ast.Slice(obj, ast.Range(None, ast.Const(T.int, 2))).set_text_ref(o.text_ref)
    table = evaluate(state, slice_ast)

    assert (table.type <= T.table), table
    rows = localize(state, table) # Must be 1 row
    if len(rows) == 0:
        if not o.nullable:
            raise Signal.make(T.ValueError, o, "'one' expected a single result, got an empty expression")
        return objects.null
    elif len(rows) > 1:
        raise Signal.make(T.ValueError, o, "'one' expected a single result, got more")

    row ,= rows
    rowtype = T.row[table.type]

    if table.type <= T.list:
        return pyvalue_inst(row)

    assert table.type <= T.table
    assert_type(table.type, T.table, o, 'one')
    d = {k: pyvalue_inst(v, table.type.elems[k], True) for k, v in row.items()}
    return objects.RowInstance(rowtype, d)


@dsp
def apply_database_rw(state: State, d: ast.Delete):
    state.catch_access(state.AccessLevels.WRITE_DB)
    # TODO Optimize: Delete on condition, not id, when possible

    cond_table = ast.Selection(d.table, d.conds).set_text_ref(d.text_ref)
    table = evaluate(state, cond_table)

    if not table.type <= T.table:
        raise Signal.make(T.TypeError, d.table, f"Expected a table. Got: {table.type}")

    if not 'name' in table.type.options:
        raise Signal.make(T.ValueError, d.table, "Cannot delete. Table is not persistent")

    rows = list(localize(state, table))
    if rows:
        if 'id' not in rows[0]:
            raise Signal.make(T.TypeError, d, "Delete error: Table does not contain id")

        ids = [row['id'] for row in rows]

        for code in sql.deletes_by_ids(table, ids):
            db_query(state, code, table.subqueries)

    return evaluate(state, d.table)

@dsp
def apply_database_rw(state: State, u: ast.Update):
    state.catch_access(state.AccessLevels.WRITE_DB)

    # TODO Optimize: Update on condition, not id, when possible
    table = evaluate(state, u.table)

    if not table.type <= T.table:
        raise Signal.make(T.TypeError, u.table, f"Expected a table. Got: {table.type}")

    if not 'name' in table.type.options:
        raise Signal.make(T.ValueError, u.table, "Cannot update: Table is not persistent")

    for f in u.fields:
        if not f.name:
            raise Signal.make(T.SyntaxError, f, f"Update requires that all fields have a name")

    # TODO verify table is concrete (i.e. lvalue, not a transitory expression)

    update_scope = {n:c for n, c in table.all_attrs().items()}
    with state.use_scope(update_scope):
        proj = {f.name:evaluate(state, f.value) for f in u.fields}

    rows = list(localize(state, table))
    if rows:
        if 'id' not in rows[0]:
            raise Signal.make(T.TypeError, u, "Update error: Table does not contain id")
        if not set(proj) < set(rows[0]):
            raise Signal.make(T.TypeError, u, "Update error: Not all keys exist in table")

        ids = [row['id'] for row in rows]

        for code in sql.updates_by_ids(table, proj, ids):
            db_query(state, code, table.subqueries)

    # TODO return by ids to maintain consistency, and skip a possibly long query
    return table


@dsp
def apply_database_rw(state: State, new: ast.NewRows):
    state.catch_access(state.AccessLevels.WRITE_DB)

    obj = state.get_var(new.type)

    if len(new.args) > 1:
        raise Signal.make(T.NotImplementedError, new, "Not yet implemented") #. Requires column-wise table concat (use join and enum)")

    if isinstance(obj, objects.UnknownInstance):
        arg ,= new.args
        table = evaluate(state, arg.value)
        fakerows = [objects.RowInstance(T.row[table], {'id': T.t_id})]
        return ast.List_(T.list[T.int], fakerows).set_text_ref(new.text_ref)

    if isinstance(obj, objects.TableInstance):
        # XXX Is it always TableInstance? Just sometimes? What's the transition here?
        obj = obj.type

    assert_type(obj, T.table, new, "'new' expected an object of type '%s', instead got '%s'")

    arg ,= new.args

    # TODO postgres can do it better!
    table = evaluate(state, arg.value)
    rows = localize(state, table)

    # TODO ensure rows are the right type

    cons = TableConstructor.make(obj)

    # TODO very inefficient, vectorize this
    ids = []
    for row in rows:
        matched = cons.match_params(state, [objects.from_python(v) for v in row.values()])
        ids += [_new_row(state, new, obj, matched).primary_key()]   # XXX return everything, not just pk?

    # XXX find a nicer way - requires a better typesystem, where id(t) < int
    return ast.List_(T.list[T.int], ids).set_text_ref(new.text_ref)


@listgen
def _destructure_param_match(state, ast_node, param_match):
    # TODO use cast rather than a ad-hoc hardwired destructure
    for k, v in param_match:
        if isinstance(v, objects.RowInstance):
            v = v.primary_key()
        v = localize(state, v)

        if k.type <= T.struct:
            names = [name for name, t in flatten_type(k.orig, [k.name])]
            if not isinstance(v, list):
                msg = f"Parameter {k.name} received a bad value: {v} (expecting a struct or a list)"
                raise Signal.make(T.TypeError, ast_node, msg)
            if len(v) != len(names):
                msg = f"Parameter {k.name} received a bad value (size of {len(names)})"
                raise Signal.make(T.TypeError, ast_node, msg)
            yield from safezip(names, v)
        else:
            yield k.name, v


def _new_value(state, v, type_):
    if isinstance(v, list):
        return evaluate(state, objects.PythonList(v))
    return objects.pyvalue_inst(v, type_=type_)

@dsp
def freeze(state, i: objects.Instance):
    return _new_value(state, cast_to_python(state, i), type_=i.type )

@dsp
def freeze(state, i: objects.RowInstance):
    return i.replace(attrs={k: freeze(state, v) for k, v in i.attrs.items()})

def _new_row(state, new_ast, table, matched):
    matched = [(k, freeze(state, evaluate(state, v))) for k, v in matched]
    destructured_pairs = _destructure_param_match(state, new_ast, matched)

    keys = [name for (name, _) in destructured_pairs]
    values = [sql.make_value(v) for (_,v) in destructured_pairs]
    assert keys and values
    # XXX use regular insert?

    if state.db.target == sql.bigquery:
        rowid = db_query(state, sql.FuncCall(T.string, 'GENERATE_UUID', []))
        keys += ['id']
        values += [sql.make_value(rowid)]
        q = sql.InsertConsts(table.options['name'].repr_name, keys, [values])
        db_query(state, q)
    else:
        q = sql.InsertConsts(table.options['name'].repr_name, keys, [values])
        # q = sql.InsertConsts(new_ast.type, keys, [values])
        db_query(state, q)
        rowid = db_query(state, sql.LastRowId())

    d = SafeDict({'id': objects.pyvalue_inst(rowid)})
    d.update({p.name:v for p, v in matched})
    return objects.RowInstance(T.row[table], d)



@dsp
def apply_database_rw(state: State, new: ast.New):
    state.catch_access(state.AccessLevels.WRITE_DB)

    obj = state.get_var(new.type)

    # XXX Assimilate this special case
    if isinstance(obj, Type) and obj <= T.Exception:
        def create_exception(state, msg):
            msg = cast_to_python(state, msg)
            assert new.text_ref is state.stacktrace[-1]
            return Signal(obj, list(state.stacktrace), msg)    # TODO move this to `throw`?
        f = objects.InternalFunction(obj.typename, [objects.Param('message')], create_exception)
        res = evaluate(state, ast.FuncCall(f, new.args).set_text_ref(new.text_ref))
        return res

    if not isinstance(obj, objects.TableInstance):
        raise Signal.make(T.TypeError, new, f"'new' expects a table or exception, instead got {obj.repr()}")

    table = obj
    # TODO assert tabletype is a real table and not a query (not transient), otherwise new is meaningless
    assert_type(table.type, T.table, new, "'new' expected an object of type '%s', instead got '%s'")

    cons = TableConstructor.make(table.type)
    matched = cons.match_params(state, new.args)

    return _new_row(state, new, table.type, matched)


@dataclass
class TableConstructor(objects.Function):
    "Serves as an ad-hoc constructor function for given table, to allow matching params"

    params: List[objects.Param]
    param_collector: Optional[objects.Param] = None
    name = 'new'

    @classmethod
    def make(cls, table):
        return cls([
            objects.Param(name, p, p.options.get('default'), orig=p).set_text_ref(getattr(name, 'text_ref', None))
            for name, p in table_params(table)
        ])


def add_as_subquery(state: State, inst: objects.Instance):
    code_cls = sql.TableName if (inst.type <= T.table) else sql.Name
    name = state.unique_name(inst)
    return inst.replace(code=code_cls(inst.code.type, name), subqueries=inst.subqueries.update({name: inst.code}))


@dsp
def resolve_parameters(state: State, x):
    return x

@dsp
def resolve_parameters(state: State, p: ast.Parameter):
    return state.get_var(p.name)


@dsp
def evaluate(state, obj: list):
    return [evaluate(state, item) for item in obj]

@dsp
def evaluate(state, obj_):
    assert context.state

    # - Generic, non-db related operations
    obj = simplify(state, obj_)
    assert obj, obj_

    if state.access_level < state.AccessLevels.COMPILE:
        return obj

    # - Compile to instances with db-specific code (sql)
    # . Compilation may fail (e.g. due to lack of DB access)
    # . Resulting code generic within the same database, and can be cached
    # obj = compile_to_inst(state.reduce_access(state.AccessLevels.COMPILE), obj)
    obj = compile_to_inst(state, obj)

    if state.access_level < state.AccessLevels.EVALUATE:
        return obj

    # - Resolve parameters to "instantiate" the cached code
    # TODO necessary?
    obj = resolve_parameters(state, obj)

    if state.access_level < state.AccessLevels.READ_DB:
        return obj

    # - Apply operations that read or write the database (delete, insert, update, one, etc.)
    obj = apply_database_rw(state, obj)

    assert not isinstance(obj, (ast.ResolveParameters, ast.ParameterizedSqlCode)), obj

    return obj


@dsp
def apply_database_rw(state, x):
    return x

#
#    localize()
# -------------
#
# Return the local value of the expression. Only requires computation if the value is an instance.
#

@dsp
def localize(state, inst: objects.AbsInstance):
    raise NotImplementedError(inst)

@dsp
def localize(state, inst: objects.AbsStructInstance):
    return {k: localize(state, evaluate(state, v)) for k, v in inst.attrs.items()}

@dsp
def localize(state, inst: objects.Instance):
    # TODO This protection doesn't work for unoptimized code
    # Cancel unoptimized mode? Or leave this unprotected?
    # state.require_access(state.AccessLevels.WRITE_DB)

    if inst.code is sql.null:
        return None

    return db_query(state, inst.code, inst.subqueries)

@dsp
def localize(state, inst: objects.ValueInstance):
    return inst.local_value

@dsp
def localize(state, inst: objects.SelectedColumnInstance):
    # XXX is this right?
    p = evaluate(state, inst.parent)
    return p.get_attr(inst.name)

@dsp
def localize(state, x):
    return x





def new_table_from_rows(state, name, columns, rows):
    # TODO check table doesn't exist

    tuples = [
        [sql.make_value(i) for i in row]
        for row in rows
    ]

    # TODO refactor into function?
    elems = {c:v.type.as_nullable() for c,v in zip(columns, tuples[0])}
    elems['id'] = T.t_id
    table = T.table(elems, temporary=True, pk=[['id']], name=Id(name))

    db_query(state, sql.compile_type_def(state, name, table))

    code = sql.InsertConsts(name, columns, tuples)
    db_query(state, code)

    x = objects.new_table(table)
    state.set_var(name, x)
    return x


def new_table_from_expr(state, name, expr, const, temporary):
    elems = expr.type.elems

    if any(t <= T.unknown for t in elems.values()):
        return objects.TableInstance.make(sql.null, expr.type, [])

    if 'id' in elems and not const:
        msg = "Field 'id' already exists. Rename it, or use 'const table' to copy it as-is."
        raise Signal.make(T.NameError, None, msg)

    table = T.table(dict(elems), name=Id(name), pk=[] if const else [['id']], temporary=temporary)

    if not const:
        table.elems['id'] = T.t_id

    db_query(state, sql.compile_type_def(state, name, table))

    read_only, flat_columns = table_flat_for_insert(table)
    expr = exclude_fields(state, expr, set(read_only) & set(elems))
    db_query(state, sql.Insert(Id(name), flat_columns, expr.code), expr.subqueries)

    return objects.new_table(table)


# cast_to_python - make sure the value is a native python object, not a preql instance

@dsp
def cast_to_python(state, obj):
    raise Signal.make(T.TypeError, None, f"Unexpected value: {pql_repr(obj.type, obj)}")

@dsp
def cast_to_python(state, obj: ast.Ast):
    inst = cast_to_instance(state, obj)
    return cast_to_python(state, inst)

@dsp
def cast_to_python(state, obj: objects.AbsInstance):
    # if state.access_level <= state.AccessLevels.QUERY:
    if obj.type <= T.projected | T.aggregated:
        raise exc.InsufficientAccessLevel(state.access_level)
        # raise Signal.make(T.CastError, None, f"Internal error. Cannot cast projected obj: {obj}")
    res = localize(state, obj)
    if obj.type == T.float:
        res = float(res)
    elif obj.type == T.int:
        res = int(res)
    elif obj.type == T.bool:
        assert res in (0, 1), res
        res = bool(res)
    return res



### Added functions

def function_localize_keys(self, state, struct):
    return cast_to_python(state, struct)

objects.Function._localize_keys = function_localize_keys


from preql.context import context
def instance_repr(self):
    return pql_repr(self.type, localize(context.state, self))

objects.Instance.repr = instance_repr

