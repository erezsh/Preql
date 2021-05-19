from typing import List, Optional
import logging
from pathlib import Path

from preql.utils import safezip, dataclass, SafeDict, listgen, method
from preql import settings
from preql.context import context

from .interp_common import assert_type, exclude_fields, call_builtin_func, is_global_scope, cast_to_python_string, cast_to_python_int
from .state import set_var, use_scope, get_var, unique_name, get_db, get_db_target, catch_access, AccessLevels, get_access_level, reduce_access
from . import exceptions as exc
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .parser import Str
from .interp_common import dsp, pyvalue_inst, cast_to_python
from .compiler import cast_to_instance
from .pql_types import T, Type, Object, Id, dp_inst
from .types_impl import table_params, table_flat_for_insert, flatten_type, pql_repr, kernel_type
from .exceptions import InsufficientAccessLevel, ReturnSignal, Signal


MODULES_PATH = Path(__file__).parent.parent / 'modules'


@dsp
def resolve(struct_def: ast.StructDef):
    members = {str(k):resolve(v) for k, v in struct_def.members}
    struct = T.struct(members)
    set_var(struct_def.name, struct)
    return struct

@dsp
def resolve(table_def: ast.TableDef):
    name = table_def.name
    if is_global_scope(context.state):
        name = get_db().qualified_name(name)
        temporary = False
    else:
        name = '__local_' + unique_name(name)
        temporary = True

    t = T.table({}, name=Id(name), temporary=temporary)

    with use_scope({table_def.name: t}):  # For self-reference
        elems = {c.name: resolve(c) for c in table_def.columns}
        t = t(elems)

    if table_def.methods:
        methods = evaluate(table_def.methods)
        t.proto_attrs.update({m.userfunc.name:m.userfunc for m in methods})

    return t

@dsp
def resolve(col_def: ast.ColumnDef):
    coltype = resolve(col_def.type)

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
def resolve(type_: ast.Type):
    t = evaluate( type_.type_obj)
    if isinstance(t, objects.TableInstance):
        t = t.type

    if not isinstance(t, (Type, objects.SelectedColumnInstance)):
        raise Signal.make(T.TypeError, type_, f"Expected type in column definition. Instead got '{t}'")

    if type_.nullable:
        t = t.as_nullable()

    return t



def db_query(sql_code, subqueries=None):
    try:
        return get_db().query(sql_code, subqueries)
    except exc.DatabaseQueryError as e:
        raise Signal.make(T.DbQueryError, None, e.args[0]) from e

def drop_table(state, table_type):
    name ,= table_type.options['name'].parts
    code = sql.compile_drop_table(name)
    return state.db.query(code, {})



@dsp
def _set_value(name: ast.Name, value):
    set_var(name.name, value)

@dsp
def _set_value(attr: ast.Attr, value):
    raise Signal.make(T.NotImplementedError, attr, f"Cannot set attribute for {attr.expr.repr()}")

def _copy_rows(target_name: ast.Name, source: objects.TableInstance):

    if source is objects.EmptyList: # Nothing to add
        return objects.null

    target = evaluate(target_name)

    params = dict(table_params(target.type))
    for p in params:
        if p not in source.type.elems:
            raise Signal.make(T.TypeError, source, f"Missing column '{p}' in {source.type}")

    primary_keys, columns = table_flat_for_insert(target.type)

    source = exclude_fields(source, set(primary_keys) & set(source.type.elems))

    code = sql.Insert(target.type.options['name'], columns, source.code)
    db_query(code, source.subqueries)
    return objects.null


@method
def _execute(struct_def: ast.StructDef):
    resolve(struct_def)

@method
def _execute(table_def: ast.TableDefFromExpr):
    state = context.state
    expr = cast_to_instance(table_def.expr)
    name = table_def.name
    if is_global_scope(state):
        temporary = False
    else:
        name = '__local_' + unique_name(name)
        temporary = True
    t = new_table_from_expr(name, expr, table_def.const, temporary)
    set_var(table_def.name, t)
    
@method
def _execute(var_def: ast.SetValue):
    res = evaluate(var_def.value)
    # res = apply_database_rw(res)
    _set_value(var_def.name, res)
    return res



@method
def _execute(table_def: ast.TableDef):
    if table_def.columns and isinstance(table_def.columns[-1], ast.Ellipsis):
        ellipsis = table_def.columns.pop()
    else:
        ellipsis = None

    if any(isinstance(c, ast.Ellipsis) for c in table_def.columns):
        # XXX why must it? just ensure it appears once
        raise Signal.make(T.SyntaxError, table_def, "Ellipsis must appear at the end")

    # Create type and a corresponding table in the database
    t = resolve(table_def)
    db_name = t.options['name']

    exists = get_db().table_exists(db_name.repr_name)
    if exists:
        assert not t.options['temporary']
        cur_type = get_db().import_table_type(db_name.repr_name, None if ellipsis else set(t.elems) | {'id'})

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

    set_var(table_def.name, inst)

    if not exists:
        sql_code = sql.compile_type_def(db_name.repr_name, t)
        db_query(sql_code)
@method
def _execute(insert_rows: ast.InsertRows):
    if not isinstance(insert_rows.name, ast.Name):
        # TODO support Attr
        raise Signal.make(T.SyntaxError, insert_rows, "L-value must be table name")

    rval = evaluate(insert_rows.value)

    assert_type(rval.type, T.table, insert_rows, '+=')

    return _copy_rows(insert_rows.name, rval)

@method
def _execute(func_def: ast.FuncDef):
    func = func_def.userfunc
    assert isinstance(func, objects.UserFunction)

    new_params = []
    for p in func.params:
        if p.type:
            t = evaluate(p.type)
            p = p.replace(type=t)
        new_params.append(p)

    set_var(func.name, func.replace(params=new_params))

@method
def _execute(p: ast.Print):
    display = context.state.display
    # TODO Can be done better. Maybe cast to ReprText?
    insts = evaluate(p.value)
    assert isinstance(insts, list)

    for inst in insts:
        if inst.type <= T.string:
            repr_ = cast_to_python_string(inst)
        else:
            repr_ = inst.repr()

        display.print(repr_, end=" ")
    display.print("")

@method
def _execute(p: ast.Assert):
    res = cast_to_python(p.cond)
    if not res:
        # TODO pretty print values
        if isinstance(p.cond, ast.Compare):
            s = (' %s '%p.cond.op).join(str(evaluate(a).repr()) for a in p.cond.args)
        else:
            s = p.cond.repr()
        raise Signal.make(T.AssertError, p.cond, f"Assertion failed: {s}")

@method
def _execute(cb: ast.CodeBlock):
    for stmt in cb.statements:
        execute(stmt)
    return objects.null


@method
def _execute(i: ast.If):
    cond = cast_to_python(i.cond)

    if cond:
        execute(i.then)
    elif i.else_:
        execute(i.else_)

@method
def _execute(w: ast.While):
    while cast_to_python(w.cond):
        execute(w.do)

@method
def _execute(f: ast.For):
    expr = cast_to_python(f.iterable)
    for i in expr:
        with use_scope({f.var: objects.from_python(i)}):
            execute(f.do)

@method
def _execute(t: ast.Try):
    try:
        execute(t.try_)
    except Signal as e:
        catch_type = evaluate(t.catch_expr).localize()
        if not isinstance(catch_type, Type):
            raise Signal.make(T.TypeError, t.catch_expr, f"Catch expected type, got {t.catch_expr.type}")
        if e.type <= catch_type:
            scope = {t.catch_name: e} if t.catch_name else {}
            with use_scope(scope):
                execute(t.catch_block)
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

    # assert state is state.interp.state    # Fix for threaded
    i = state.interp.clone(use_core=r.use_core)

    state.stacktrace.append(r.text_ref)
    try:
        i.include(module_path)
    finally:
        assert state.stacktrace[-1] is r.text_ref
        state.stacktrace.pop()

    # Inherit module db (in case it called connect())
    assert state.db is i.state.db

    ns = i.state.ns
    assert len(ns) == 1
    return objects.Module(r.module_path, ns._ns[0])


@method
def _execute(r: ast.Import):
    module = import_module(context.state, r)
    set_var(r.as_name or r.module_path, module)
    return module

@method
def _execute(r: ast.Return):
    value = evaluate(r.value)
    raise ReturnSignal(value)

@method
def _execute(t: ast.Throw):
    e = evaluate(t.value)
    if isinstance(e, ast.Ast):
        raise exc.InsufficientAccessLevel()
    assert isinstance(e, Exception), e
    raise e

def execute(stmt):
    if isinstance(stmt, ast.Statement):
        return stmt._execute() or objects.null
    return evaluate(stmt)




# Simplify performs local operations before any db-specific compilation occurs
# Technically not super useful at the moment, but makes conceptual sense.

@method
def simplify(cb: ast.CodeBlock):
    # if len(cb.statements) == 1:
    #     s ,= cb.statements
    #     return simplify(s)
    try:
        return cb._execute()
    except ReturnSignal as r:
        # XXX is this correct?
        return r.value
    except Signal as e:
        # Failed to run it, so try to cast as instance
        # XXX order should be other way around!
        if e.type <= T.CastError:
            return cb.compile_to_inst()
        raise
    except InsufficientAccessLevel:
        return cb

@method
def simplify(n: ast.Name):
    # XXX what happens to caching if this is a global variable?
    return get_var(n.name)

@method
def simplify(x: Object):
    return x

# @dsp
# def simplify(ls: list):
#     return [simplify(i) for i in ls]

# @dsp
# def simplify(d: objects.ParamDict):
#     return d.replace(params={name: evaluate( v) for name, v in d.params.items()})

# @dsp
# def simplify(node: ast.Ast):
#     # return _simplify_ast(state, node)
#     return node

# def _simplify_ast(state, node):
#     resolved = {k:simplify(v) for k, v in node
#                 if isinstance(v, types.PqlObject) or isinstance(v, list) and all(isinstance(i, types.PqlObject) for i in v)}
#     return node.replace(**resolved)

# TODO isn't this needed somewhere??
# @dsp
# def simplify(if_: ast.If):
#     if_ = _simplify_ast(state, if_)
#     if isinstance(if_.cond, objects.ValueInstance): # XXX a more general test?
#         if if_.cond.local_value:
#             return if_.then
#         else:
#             return if_.else_
#     return if_


# TODO Optimize these, right now failure to evaluate will lose all work
@method
def simplify(obj: ast.Or):
    a, b = evaluate( obj.args)
    ta = kernel_type(a.type)
    tb = kernel_type(b.type)
    if ta != tb:
        raise Signal.make(T.TypeError, obj, f"'or' operator requires both arguments to be of the same type, but got '{ta}' and '{tb}'.")
    try:
        if a.test_nonzero():
            return a
    except InsufficientAccessLevel:
        return obj
    return b


@method
def simplify(obj: ast.And):
    a, b = evaluate( obj.args)
    ta = kernel_type(a.type)
    tb = kernel_type(b.type)
    if ta != tb:
        raise Signal.make(T.TypeError, obj, f"'and' operator requires both arguments to be of the same type, but got '{ta}' and '{tb}'.")
    try:
        if not a.test_nonzero():
            return a
    except InsufficientAccessLevel:
        return obj
    return b


@method
def simplify(obj: ast.Not):
    inst = evaluate( obj.expr)
    try:
        nz = inst.test_nonzero()
    except InsufficientAccessLevel:
        return obj
    return objects.pyvalue_inst(not nz)



@method
def simplify(funccall: ast.FuncCall):
    state = context.state
    func = evaluate(funccall.func)

    if isinstance(func, objects.UnknownInstance):
        # evaluate( [a.value for a in funccall.args])
        raise Signal.make(T.TypeError, funccall.func, f"Error: Object of type '{func.type}' is not callable")

    args = funccall.args
    if isinstance(func, Type):
        # Cast to type
        args = args + [func]
        func = get_var('cast')

    if not isinstance(func, objects.Function):
        raise Signal.make(T.TypeError, funccall.func, f"Error: Object of type '{func.type}' is not callable")

    state.stacktrace.append(funccall.text_ref)
    try:
        res = eval_func_call(func, args)
    finally:
        assert state.stacktrace[-1] is funccall.text_ref
        state.stacktrace.pop()

    assert isinstance(res, Object), (type(res), res)
    return res


def eval_func_call(func, args):
    state = context.state
    assert isinstance(func, objects.Function)

    matched_args = func.match_params(args)

    if isinstance(func, objects.MethodInstance):
        ordered_args = {'this': func.parent}
        func = func.func
        # args.update(func.parent.all_attrs())
    else:
        ordered_args = {}

    # XXX simplify destroys text_ref, so it harms error messages.
    # TODO Can I get rid of it, or make it preserve the text_ref somehow?
    # Don't I need an instance to ensure I have type?

    for i, (p, a) in enumerate(matched_args):
        if not p.name.startswith('$'):      # $param means don't evaluate expression, leave it to the function
            a = evaluate( a)
        # TODO cast?
        if p.type and not a.type <= p.type:
            raise Signal.make(T.TypeError, func, f"Argument #{i} of '{func.name}' is of type '{a.type}', expected '{p.type}'")
        ordered_args[p.name] = a


    if isinstance(func, objects.InternalFunction):
        # TODO ensure pure function?
        # TODO Ensure correct types
        ordered_args = list(ordered_args.values())
        return func.func(*ordered_args)

    # TODO make tests to ensure caching was successful
    expr = func.expr
    if settings.cache:
        params = {name: ast.Parameter(name, value.type) for name, value in ordered_args.items()}
        sig = (func.name,) + tuple(a.type for a in ordered_args.values())

        try:
            with use_scope(params):
                if sig in state._cache:
                    compiled_expr = state._cache[sig]
                else:
                    logging.info(f"Compiling.. {func}")
                    with context(state=reduce_access(AccessLevels.COMPILE)):
                        compiled_expr = _call_expr(func.expr)
                    logging.info("Compiled successfully")
                    if isinstance(compiled_expr, objects.Instance):
                        # XXX a little ugly
                        qb = sql.QueryBuilder(get_db_target(), True)
                        x = compiled_expr.code.compile(qb)
                        x = x.optimize()
                        compiled_expr = compiled_expr.replace(code=x)
                    state._cache[sig] = compiled_expr

            expr = ast.ResolveParameters(compiled_expr, ordered_args)

        except exc.InsufficientAccessLevel:
            # Don't cache
            pass

    with use_scope(ordered_args):
        res = _call_expr(expr)

    if isinstance(res, ast.ResolveParameters):  # XXX A bit of a hack
        raise exc.InsufficientAccessLevel()

    return res


def _call_expr(expr):
    try:
        return evaluate(expr)
    except ReturnSignal as r:
        return r.value

# TODO fix these once we have proper types
@method
def test_nonzero(table: objects.TableInstance):
    count = call_builtin_func("count", [table])
    return bool(cast_to_python_int(count))

@method
def test_nonzero(inst: objects.Instance):
    return bool(cast_to_python(inst))

@method
def test_nonzero(inst: Type):
    return True








@method
def apply_database_rw(o: ast.One):
    # TODO move these to the core/base module
    obj = evaluate( o.expr)
    if obj.type <= T.struct:
        if len(obj.attrs) != 1:
            raise Signal.make(T.ValueError, o, f"'one' expected a struct with a single attribute, got {len(obj.attrs)}")
        x ,= obj.attrs.values()
        return x

    slice_ast = ast.Slice(obj, ast.Range(None, ast.Const(T.int, 2))).set_text_ref(o.text_ref)
    table = evaluate( slice_ast)

    assert (table.type <= T.table), table
    rows = table.localize() # Must be 1 row
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


@method
def apply_database_rw(d: ast.Delete):
    catch_access(AccessLevels.WRITE_DB)
    # TODO Optimize: Delete on condition, not id, when possible

    cond_table = ast.Selection(d.table, d.conds).set_text_ref(d.text_ref)
    table = evaluate( cond_table)

    if not table.type <= T.table:
        raise Signal.make(T.TypeError, d.table, f"Expected a table. Got: {table.type}")

    if not 'name' in table.type.options:
        raise Signal.make(T.ValueError, d.table, "Cannot delete. Table is not persistent")

    rows = list(table.localize())
    if rows:
        if 'id' not in rows[0]:
            raise Signal.make(T.TypeError, d, "Delete error: Table does not contain id")

        ids = [row['id'] for row in rows]

        for code in sql.deletes_by_ids(table, ids):
            db_query(code, table.subqueries)

    return evaluate( d.table)

@method
def apply_database_rw(u: ast.Update):
    catch_access(AccessLevels.WRITE_DB)

    # TODO Optimize: Update on condition, not id, when possible
    table = evaluate( u.table)

    if not table.type <= T.table:
        raise Signal.make(T.TypeError, u.table, f"Expected a table. Got: {table.type}")

    if not 'name' in table.type.options:
        raise Signal.make(T.ValueError, u.table, "Cannot update: Table is not persistent")

    for f in u.fields:
        if not f.name:
            raise Signal.make(T.SyntaxError, f, f"Update requires that all fields have a name")

    # TODO verify table is concrete (i.e. lvalue, not a transitory expression)

    update_scope = {n:c for n, c in table.all_attrs().items()}
    with use_scope(update_scope):
        proj = {f.name:evaluate( f.value) for f in u.fields}

    rows = list(table.localize())
    if rows:
        if 'id' not in rows[0]:
            raise Signal.make(T.TypeError, u, "Update error: Table does not contain id")
        if not set(proj) < set(rows[0]):
            raise Signal.make(T.TypeError, u, "Update error: Not all keys exist in table")

        ids = [row['id'] for row in rows]

        for code in sql.updates_by_ids(table, proj, ids):
            db_query(code, table.subqueries)

    # TODO return by ids to maintain consistency, and skip a possibly long query
    return table


@method
def apply_database_rw(new: ast.NewRows):
    catch_access(AccessLevels.WRITE_DB)

    obj = get_var(new.type)

    if len(new.args) > 1:
        raise Signal.make(T.NotImplementedError, new, "Not yet implemented") #. Requires column-wise table concat (use join and enum)")

    if isinstance(obj, objects.UnknownInstance):
        arg ,= new.args
        table = evaluate( arg.value)
        fakerows = [objects.RowInstance(T.row[table], {'id': T.t_id})]
        return ast.List_(T.list[T.int], fakerows).set_text_ref(new.text_ref)

    if isinstance(obj, objects.TableInstance):
        # XXX Is it always TableInstance? Just sometimes? What's the transition here?
        obj = obj.type

    assert_type(obj, T.table, new, "'new' expected an object of type '%s', instead got '%s'")

    arg ,= new.args

    # TODO postgres can do it better!
    table = evaluate( arg.value)
    rows = table.localize()

    # TODO ensure rows are the right type

    cons = TableConstructor.make(obj)

    # TODO very inefficient, vectorize this
    ids = []
    for row in rows:
        matched = cons.match_params([objects.from_python(v) for v in row.values()])
        ids += [_new_row(new, obj, matched).primary_key()]   # XXX return everything, not just pk?

    # XXX find a nicer way - requires a better typesystem, where id(t) < int
    return ast.List_(T.list[T.int], ids).set_text_ref(new.text_ref)


@listgen
def _destructure_param_match(ast_node, param_match):
    # TODO use cast rather than a ad-hoc hardwired destructure
    for k, v in param_match:
        if isinstance(v, objects.RowInstance):
            v = v.primary_key()
        v = v.localize()

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


def _new_value(v, type_):
    if isinstance(v, list):
        return evaluate( objects.PythonList(v))
    return objects.pyvalue_inst(v, type_=type_)

@dsp
def freeze(i: objects.Instance):
    return _new_value(cast_to_python(i), type_=i.type )

@dsp
def freeze(i: objects.RowInstance):
    return i.replace(attrs={k: freeze(v) for k, v in i.attrs.items()})

def _new_row(new_ast, table, matched):
    matched = [(k, freeze(evaluate( v))) for k, v in matched]
    destructured_pairs = _destructure_param_match(new_ast, matched)

    keys = [name for (name, _) in destructured_pairs]
    values = [sql.make_value(v) for (_,v) in destructured_pairs]
    assert keys and values
    # XXX use regular insert?

    if 'name' not in table.options:
        raise Signal.make(T.TypeError, new_ast, f"'new' expects a persistent table. Instead got a table expression.")

    if get_db_target() == sql.bigquery:
        rowid = db_query(sql.FuncCall(T.string, 'GENERATE_UUID', []))
        keys += ['id']
        values += [sql.make_value(rowid)]
        q = sql.InsertConsts(table.options['name'].repr_name, keys, [values])
        db_query(q)
    else:
        q = sql.InsertConsts(table.options['name'].repr_name, keys, [values])
        # q = sql.InsertConsts(new_ast.type, keys, [values])
        db_query(q)
        rowid = db_query(sql.LastRowId())

    d = SafeDict({'id': objects.pyvalue_inst(rowid)})
    d.update({p.name:v for p, v in matched})
    return objects.RowInstance(T.row[table], d)



@method
def apply_database_rw(new: ast.New):
    catch_access(AccessLevels.WRITE_DB)

    obj = get_var(new.type)

    # XXX Assimilate this special case
    if isinstance(obj, Type) and obj <= T.Exception:
        def create_exception(msg):
            state = context.state
            msg = cast_to_python(msg)
            assert new.text_ref is state.stacktrace[-1]
            return Signal(obj, list(state.stacktrace), msg)    # TODO move this to `throw`?
        f = objects.InternalFunction(obj.typename, [objects.Param('message')], create_exception)
        res = evaluate( ast.FuncCall(f, new.args).set_text_ref(new.text_ref))
        return res

    if not isinstance(obj, objects.TableInstance):
        raise Signal.make(T.TypeError, new, f"'new' expects a table or exception, instead got {obj.repr()}")

    table = obj
    # TODO assert tabletype is a real table and not a query (not transient), otherwise new is meaningless
    assert_type(table.type, T.table, new, "'new' expected an object of type '%s', instead got '%s'")

    cons = TableConstructor.make(table.type)
    matched = cons.match_params(new.args)

    return _new_row(new, table.type, matched)

@method
def apply_database_rw(x: Object):
    return x


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


def add_as_subquery(inst: objects.Instance):
    code_cls = sql.TableName if (inst.type <= T.table) else sql.Name
    name = unique_name(inst)
    return inst.replace(code=code_cls(inst.code.type, name), subqueries=inst.subqueries.update({name: inst.code}))


@method
def resolve_parameters(x: Object):
    return x

@method
def resolve_parameters(p: ast.Parameter):
    return get_var(p.name)


@dsp
def evaluate(obj: list):
    return [evaluate( item) for item in obj]

@dsp
def evaluate(obj_):
    access_level = get_access_level()

    # - Generic, non-db related operations
    obj = obj_.simplify()
    assert obj, obj_

    if access_level < AccessLevels.COMPILE:
        return obj

    # - Compile to instances with db-specific code (sql)
    # . Compilation may fail (e.g. due to lack of DB access)
    # . Resulting code generic within the same database, and can be cached
    # obj = compile_to_inst(state.reduce_access(state.AccessLevels.COMPILE), obj)
    obj = obj.compile_to_inst()

    if access_level < AccessLevels.EVALUATE:
        return obj

    # - Resolve parameters to "instantiate" the cached code
    # TODO necessary?
    obj = obj.resolve_parameters()

    if access_level < AccessLevels.READ_DB:
        return obj

    # - Apply operations that read or write the database (delete, insert, update, one, etc.)
    obj = obj.apply_database_rw()

    assert not isinstance(obj, (ast.ResolveParameters, ast.ParameterizedSqlCode)), obj

    return obj



#
#    localize()
# -------------
#
# Return the local value of the expression. Only requires computation if the value is an instance.
#

@method
def localize(inst: objects.AbsInstance):
    raise NotImplementedError(inst)

@method
def localize(inst: objects.AbsStructInstance):
    return {k: evaluate(v).localize() for k, v in inst.attrs.items()}

@method
def localize(inst: objects.Instance):
    # TODO This protection doesn't work for unoptimized code
    # Cancel unoptimized mode? Or leave this unprotected?
    # state.require_access(state.AccessLevels.WRITE_DB)

    if inst.code is sql.null:
        return None

    return db_query(inst.code, inst.subqueries)

@method
def localize(inst: objects.ValueInstance):
    return inst.local_value

@method
def localize(inst: objects.SelectedColumnInstance):
    # XXX is this right?
    p = evaluate(inst.parent)
    return p.get_attr(inst.name)

@method
def localize(x: Object):
    return x





def new_table_from_rows(name, columns, rows):
    # TODO check table doesn't exist

    tuples = [
        [sql.make_value(i) for i in row]
        for row in rows
    ]

    # TODO refactor into function?
    elems = {c:v.type.as_nullable() for c,v in zip(columns, tuples[0])}
    elems['id'] = T.t_id
    table = T.table(elems, temporary=True, pk=[['id']], name=Id(name))

    db_query(sql.compile_type_def(name, table))

    code = sql.InsertConsts(name, columns, tuples)
    db_query(code)

    x = objects.new_table(table)
    set_var(name, x)
    return x


def new_table_from_expr(name, expr, const, temporary):
    elems = expr.type.elems

    if any(t <= T.unknown for t in elems.values()):
        return objects.TableInstance.make(sql.null, expr.type, [])

    if 'id' in elems and not const:
        msg = "Field 'id' already exists. Rename it, or use 'const table' to copy it as-is."
        raise Signal.make(T.NameError, None, msg)

    table = T.table(dict(elems), name=Id(name), pk=[] if const else [['id']], temporary=temporary)

    if not const:
        table.elems['id'] = T.t_id

    db_query(sql.compile_type_def(name, table))

    read_only, flat_columns = table_flat_for_insert(table)
    expr = exclude_fields(expr, set(read_only) & set(elems))
    db_query(sql.Insert(Id(name), flat_columns, expr.code), expr.subqueries)

    return objects.new_table(table)


# cast_to_python - make sure the value is a native python object, not a preql instance

@dsp
def cast_to_python(obj):
    raise Signal.make(T.TypeError, None, f"Unexpected value: {pql_repr(obj.type, obj)}")

@dsp
def cast_to_python(obj: ast.Ast):
    inst = cast_to_instance(obj)
    return cast_to_python(inst)

@dsp
def cast_to_python(obj: objects.AbsInstance):
    # if state.access_level <= state.AccessLevels.QUERY:
    if obj.type <= T.projected | T.aggregated:
        raise exc.InsufficientAccessLevel(get_access_level())
        # raise Signal.make(T.CastError, None, f"Internal error. Cannot cast projected obj: {obj}")
    res = obj.localize()
    if obj.type == T.float:
        res = float(res)
    elif obj.type == T.int:
        res = int(res)
    elif obj.type == T.bool:
        assert res in (0, 1), res
        res = bool(res)
    return res



### Added functions

def function_localize_keys(self, struct):
    return cast_to_python(struct)

objects.Function._localize_keys = function_localize_keys


def instance_repr(self):
    return pql_repr(self.type, self.localize())

objects.Instance.repr = instance_repr


@dp_inst
def post_instance_getattr(inst, p: T.property):
    return eval_func_call(objects.MethodInstance(inst, p.func), [])
