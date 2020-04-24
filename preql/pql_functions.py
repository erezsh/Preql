import inspect
from copy import copy
from typing import Optional

from .utils import safezip, listgen, SafeDict
from .exceptions import pql_TypeError, pql_JoinError, pql_ValueError, pql_ExitInterp

from . import pql_objects as objects
from . import pql_types as types
from . import pql_ast as ast
from . import sql

from .compiler import compile_type_def
from .interp_common import State, new_value_instance, dy, exclude_fields
from .evaluate import evaluate, localize

# def _pql_SQL_callback(state: State, var: str, instances):
#     var = var.group()
#     assert var[0] == '$'
#     var_name = var[1:]
#     obj = state.get_var(var_name)

#     if isinstance(obj, types.TableType):
#         # This branch isn't strictly necessary
#         # It exists to create nicer SQL code output
#         inst = objects.TableInstance.make(sql.TableName(obj, obj.name), obj, [], {})
#     else:
#         inst = evaluate(state, obj)

#         # if isinstance(inst, objects.TableInstance):

#         #     # Make new type
#         #     new_columns = {
#         #         name: objects.make_column_instance(sql.Name(col.type, name), col.type, [col])
#         #         for name, col in inst.columns.items()
#         #     }

#         #     # Make code
#         #      ql_fields = [
#         #         sql.ColumnAlias.make(o.code, n.code)
#         #          or old, new in safezip(inst.columns.values(), new_columns.values())
#         #         for o, n in safezip(old.flatten(), new.flatten())
#         #

#         #     code = sql.Select(inst.type, inst.code, sql_fields)

#         #     # Make Instance
#         #     inst = objects.TableInstance.make(code, inst.type, [inst], new_columns)

#     instances.append(inst)

#     qb = sql.QueryBuilder(state.db.target, False)
#     return '%s' % inst.code.compile(qb).text



def _pql_PY_callback(state: State, var: str):
    var = var.group()
    assert var[0] == '$'
    var_name = var[1:]
    obj = state.get_var(var_name)
    inst = evaluate(state, obj)

    if not isinstance(inst, objects.ValueInstance):
        raise pql_TypeError(None, f"Cannot convert {inst} to a Python value")

    return '%s' % (inst.local_value)

def pql_PY(state: State, code_expr: ast.Expr):
    code_expr2 = evaluate(state, code_expr)
    py_code = localize(state, code_expr2)

    py_code = re.sub(r"\$\w+", lambda m: _pql_PY_callback(state, m), py_code)

    try:
        res = eval(py_code)
    except Exception as e:
        raise pql_ValueError(code_expr.meta, f"Python code provided returned an error: {e}")
    return objects.new_value_instance(res)


import re
def pql_SQL(state: State, type_expr: ast.Expr, code_expr: ast.Expr):
    # TODO optimize for when the string is known (prefetch the variables and return Sql)
    type_ = evaluate(state, type_expr)
    code_expr2 = evaluate(state, code_expr)
    return ast.ResolveParametersString(None, type_, code_expr2)


    # return
    # assert isinstance(code_expr, str), code_expr   # Otherwise requires to snapshot the namespace
    # embedded_vars = re.findall(r"\$(\w+)", code_expr)
    # return ast.ResolveParametersString(code_expr.meta, type_, code_expr, {name: state.get_var(name) for name in embedded_vars})
    # return

    # type_ = simplify(state, type_expr)
    # sql_code = localize(state, evaluate(state, code_expr))
    # assert isinstance(sql_code, str)

    # # TODO escaping for security?
    # instances = []

    # expanded = re.sub(r"\$\w+", lambda m: _pql_SQL_callback(state, m, instances), sql_code)
    # code = sql.RawSql(type_, expanded)
    # # code = sql.ResolveParameters(sql_code)

    # # TODO validation!!
    # if isinstance(type_, types.TableType):
    #     name = get_alias(state, "subq_")

    #     inst = instanciate_table(state, type_, sql.TableName(type_, name), instances)
    #     # TODO this isn't in the tests!
    #     fields = [sql.Name(c, path) for path, c in inst.type.flatten_type()]

    #     subq = sql.Subquery(name, fields, code)
    #     inst.subqueries[name] = subq

    #     return inst

    # return objects.Instance.make(code, type_, instances)

import inspect
def _canonize_default(d):
    return None if d is inspect._empty else d

def create_internal_func(fname, f):
    sig = inspect.signature(f)
    return objects.InternalFunction(fname, [
        objects.Param(None, pname, types.any_t, _canonize_default(sig.parameters[pname].default))
        for pname, type_ in list(f.__annotations__.items())[1:]
    ], f)


def pql_brk_continue(state):
    "Continue the execution of the code (exit debug interpreter)"
    pql_exit(state, objects.null)


def pql_breakpoint(state: State):
    breakpoint()
    return objects.null

def pql_debug(state: State):
    "Hop into a debug session with REPL"
    with state.use_scope(breakpoint_funcs):
        state._py_api.start_repl('debug> ')
    return objects.null


@dy
def _pql_issubclass(a, b):
    return False

@dy
def _pql_issubclass(a: types.Primitive, b: types.ListType):
    return False

@dy
def _pql_issubclass(a, b: types.AnyType):
    assert b is types.any_t
    return True

@dy
def _pql_issubclass(a: types.ListType, b: types.ListType):
    return _pql_issubclass(a.elemtype, b.elemtype)

@dy
def _pql_issubclass(a: types.Aggregated, b: types.Aggregated):
    return _pql_issubclass(a.elemtype, b.elemtype)


def pql_isa(state: State, expr: ast.Expr, type_expr: ast.Expr):
    "Returns whether the give object is an instance of the given type"
    inst = evaluate(state, expr)
    type_ = evaluate(state, type_expr)
    # res = isinstance(inst.type, type_)
    res = inst.isa(type_)
    # res = _pql_issubclass(inst.type, type_)
    return new_value_instance(res, types.Bool)

def _count(state, obj: ast.Expr, table_func, name):
    obj = evaluate(state, obj)

    if isinstance(obj.type, types.Collection):
        code = table_func(obj.code)
    else:
        if not isinstance(obj.type, types.Aggregated):
            raise pql_TypeError(None, f"Function '{name}' expected an aggregated list, but got '{obj.type}' instead. Did you forget to group?")

        obj = obj.primary_key()
        code = sql.FieldFunc(name, obj.code)

    return objects.Instance.make(code, types.Int, [obj])

def pql_count(state: State, obj: ast.Expr):
    "Count how many rows are in the given table, or in the projected column."
    return _count(state, obj, sql.CountTable, 'count')


def pql_temptable(state: State, expr_ast: ast.Expr, const: objects = objects.null):
    """Generate a temporary table with the contents of the given table

    It will remain available until the db-session ends, unless manually removed.
    """
    # 'temptable' creates its own counting 'id' field. Copying existing 'id' fields will cause a collision
    # 'const temptable' doesn't
    expr = evaluate(state, expr_ast)
    const = localize(state, const)
    assert isinstance(expr.type, types.Collection), expr

    name = state.unique_name("temp_" + expr.type.name)
    columns = dict(expr.type.columns)


    if 'id' in columns and not const:
            raise pql_ValueError(None, "Field 'id' already exists. Rename it, or use 'const temptable' to copy it as-is.")

    table = types.TableType(name, SafeDict(columns), temporary=True, primary_keys=[['id']] if 'id' in columns else [], autocount=[] if const else ['id'])

    if not const:
        table.columns['id'] = types.IdType(table)

    state.db.query(compile_type_def(state, table))

    read_only, flat_columns = table.flat_for_insert()
    expr = exclude_fields(state, expr, read_only)
    state.db.query(sql.Insert(table, flat_columns, expr.code), expr.subqueries)

    return objects.new_table(table)

def pql_get_db_type(state: State):
    """
    Returns a string representing the type of the active database.

    Possible values are:
        - "sqlite"
        - "postgres"
    """
    assert state.access_level >= state.AccessLevels.EVALUATE
    return new_value_instance(state.db.target, types.String)



def sql_bin_op(state, op, table1, table2, name):
    t1 = evaluate(state, table1)
    t2 = evaluate(state, table2)
    # TODO make sure both table types are compatiable
    l1 = len(t1.type.flatten_type())
    l2 = len(t2.type.flatten_type())
    if l1 != l2:
        raise pql_TypeError(None, f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    code = sql.TableArith(op, [t1.code, t2.code])
    # TODO new type, so it won't look like the physical table
    return objects.TableInstance.make(code, t1.type, [t1, t2])

def pql_intersect(state, t1, t2):
    "Intersect two tables"
    return sql_bin_op(state, "INTERSECT", t1, t2, "intersect")

def pql_substract(state, t1, t2):
    "Substract two tables (except)"
    return sql_bin_op(state, "EXCEPT", t1, t2, "substract")

def pql_union(state, t1, t2):
    "Union two tables"
    return sql_bin_op(state, "UNION", t1, t2, "union")

def pql_concat(state, t1, t2):
    "Concatenate two tables (union all)"
    return sql_bin_op(state, "UNION ALL", t1, t2, "concatenate")



def _join(state: State, join: str, exprs: dict, joinall=False, nullable=None):
    assert len(exprs) == 2
    exprs = {name: evaluate(state, value) for name,value in exprs.items()}
    assert all(isinstance(x, objects.AbsInstance) for x in exprs.values())

    (a,b) = exprs.values()

    if a is objects.EmptyList or b is objects.EmptyList:
        raise pql_TypeError(None, "Cannot join on an untyped empty list")

    if isinstance(a, objects.AttrInstance) and isinstance(b, objects.AttrInstance):
        cols = a, b
        tables = [a.parent, b.parent]
    else:
        if not (isinstance(a.type, types.Collection) and isinstance(b.type, types.Collection)):
            raise pql_TypeError(None, f"join() got unexpected values:\n * {a}\n * {b}")
        if joinall:
            tables = (a, b)
        else:
            cols = _auto_join(state, join, a, b)
            tables = [c.parent for c in cols]

    assert all(isinstance(t.type, types.Collection) for t in tables)

    structs = {name: table.type.to_struct_type() for name, table in safezip(exprs, tables)}

    # Update nullable for left/right/outer joins
    if nullable:
        structs = {name: types.OptionalType(t) if n else t
                    for (name, t), n in safezip(structs.items(), nullable)}

    tables = [objects.aliased_table(t, n) for n, t in safezip(exprs, tables)]

    primary_keys = [ [name] + pk
                        for name, t in safezip(exprs, tables)
                        for pk in t.type.primary_keys
                    ]
    table_type = types.TableType(state.unique_name("joinall" if joinall else "join"), SafeDict(structs), False, primary_keys)

    conds = [] if joinall else [sql.Compare('=', [sql.Name(c.type, types.join_names((n, c.name))) for n, c in safezip(structs, cols)])]

    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    return objects.TableInstance.make(code, table_type, [a,b])

def pql_join(state, tables):
    "Inner join two tables into a new projection {t1, t2}"
    return _join(state, "JOIN", tables)
def pql_leftjoin(state, tables):
    "Left join two tables into a new projection {t1, t2}"
    return _join(state, "LEFT JOIN", tables, nullable=[False, True])
def pql_joinall(state: State, tables):
    "Cartesian product of two tables into a new projection {t1, t2}"
    return _join(state, "JOIN", tables, True)

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
    for name, c in t1.type.columns.items():
        if isinstance(c, types.RelationalColumn):
            rel = c.type
            if rel == t2.type:
                # TODO depends on the query XXX
                yield (objects.AttrInstance(t2, types.IdType(t2.type), 'id'), objects.AttrInstance(t1, c.col_type, name))

def pql_type(state: State, obj: ast.Expr):
    """
    Returns the type of the given object
    """
    inst = evaluate(state, obj)
    t = inst.type   # XXX concrete?
    return t

def pql_cast(state: State, obj: ast.Expr, type_: ast.Expr):
    "Attempt to cast an object to a specified type"
    inst = evaluate(state, obj)
    type_ = evaluate(state, type_)
    if not isinstance(type_, types.PqlType):
        raise pql_TypeError(type_.meta, f"Cast expected a type, got {type_} instead.")

    return _cast(state, inst.type, type_, inst)

@dy
def _cast(state, inst_type: types.ListType, target_type: types.ListType, inst):
    if inst_type == target_type:
        return inst

    if inst is objects.EmptyList:
        # return inst.replace(type=target_type)
        return evaluate(state, ast.List_( None, target_type, []), )

    raise pql_TypeError(None, "Cast not fully implemented yet")


def pql_cast_int(state: State, expr: ast.Expr):
    "Temporary function, for internal use"
    # TODO make this function less ugly and better for the user
    inst = evaluate(state, expr)
    type_ = inst.type
    if isinstance(type_, types.TableType):
        assert len(inst.type.columns) == 1
        res = localize(state, inst)
        # res = types.Int.import_result(res)
        assert len(res) == 1
        res = list(res[0].values())[0]
        return new_value_instance(res, types.Int)

    elif type_ == types.Int:
        return inst
    elif isinstance(type_, types.IdType):
        return inst.replace(type=types.Int)
    elif type_ == types.Float:
        value = localize(state, inst)
        return new_value_instance(int(value), types.Int)

    raise pql_TypeError(expr.meta, f"Cannot cast expr of type {inst.type} to int")

def pql_connect(state: State, uri: ast.Expr):
    """
    Connect to a new database, specified by the uri
    """
    uri = localize(state, evaluate(state, uri))
    state.connect(uri)
    return objects.null

def pql_help(state: State, obj: types.PqlObject = objects.null):
    """
    Provides a brief summary for a given object
    """
    if obj is objects.null:
        text = (
            "Welcome to Preql!\n\n"
            "To see the list of functions and objects available in the namespace, type 'ls()'\n\n"
            "To get help for a specific function, type 'help(func_object)'\n\n"
            "For example:\n"
            "    >> help(help)\n"
        )
        return new_value_instance(text, types.String)


    lines = []
    inst = evaluate(state, obj)
    if isinstance(inst, objects.Function):
        # lines += [
        #     f"Function {inst.name}, accepts {len(inst.params)} parameters:"
        # ]
        # for p in inst.params:
        #     if p.default is not None:
        #         lines += [ f"    - {p.name}: {p.type} = {localize(state, evaluate(state, p.default))}" ]
        #     else:
        #         lines += [ f"    - {p.name}: {p.type}" ]
        param_str = ', '.join(p.name if p.default is None else f'{p.name}={localize(state, evaluate(state, p.default))}' for p in inst.params)
        if inst.param_collector is not None:
            param_str += ", ...keyword_args"
        lines = ['', f"func {inst.name}({param_str})",'']
        if isinstance(inst, objects.InternalFunction) and inst.func.__doc__:
            lines += [inst.func.__doc__]
    else:
        raise pql_TypeError(None, "help() only accepts functions at the moment")

    text = '\n'.join(lines) + '\n'
    return new_value_instance(text).replace(type=types.Text)

def pql_ls(state: State, obj: types.PqlObject = objects.null):
    """
    List all names in the namespace of the given object.

    If no object is given, lists the names in the current namespace.
    """
    # TODO support all objects
    if obj is not objects.null:
        inst = evaluate(state, obj)
        if not isinstance(inst.type, types.Collection): # XXX temp.
            raise pql_TypeError(obj.meta, "Argument to ls() must be a table")
        all_vars = list(inst.all_attrs())
    else:
        all_vars = list(state.ns.get_all_vars())

    assert all(isinstance(s, str) for s in all_vars)
    names = [new_value_instance(str(s), types.String) for s in all_vars]
    return evaluate(state, ast.List_(None, types.ListType(types.String), names))



def create_internal_funcs(d):
    new_d = {}
    for names, f in d.items():
        if isinstance(names, str):
            names = (names,)
        for name in names:
            new_d[name] = create_internal_func(name, f)
    return new_d

breakpoint_funcs = create_internal_funcs({
    ('c', 'continue'): pql_brk_continue
})


def pql_exit(state, value: types.object_t = None):
    """Exit the current interpreter instance.

    Can be used from running code, or the REPL.

    If the current interpreter is nested within another Preql interpreter (e.g. by using debug()),
    exit() will return to the parent interpreter.
    """
    raise pql_ExitInterp(value)


internal_funcs = create_internal_funcs({
    'exit': pql_exit,
    'help': pql_help,
    'ls': pql_ls,
    'connect': pql_connect,
    'count': pql_count,
    'temptable': pql_temptable,
    'concat': pql_concat,
    'intersect': pql_intersect,
    'union': pql_union,
    'substract': pql_substract,
    'SQL': pql_SQL,
    'PY': pql_PY,
    'isa': pql_isa,
    'type': pql_type,
    'debug': pql_debug,
    '_breakpoint': pql_breakpoint,
    'get_db_type': pql_get_db_type,
    '_cast_int': pql_cast_int,
    'cast': pql_cast,
})

joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param(None, 'tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param(None, 'tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param(None, 'tables')),
}
