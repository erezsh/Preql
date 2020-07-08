import inspect
import re
import csv
import inspect
from typing import Optional

from tqdm import tqdm

from .utils import safezip, listgen
from .exceptions import pql_TypeError, pql_JoinError, pql_ValueError, pql_ExitInterp

from . import pql_objects as objects
from . import pql_ast as ast
from . import sql

from .interp_common import State, new_value_instance, dy, exclude_fields, assert_type
from .evaluate import evaluate, localize, db_query, TableConstructor
from .pql_types import Object, T, table_flat_for_insert, Type, join_names, combined_dp

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
        raise pql_TypeError.make(state, None, f"Cannot convert {inst} to a Python value")

    return '%s' % (inst.local_value)

def pql_PY(state: State, code_expr: ast.Expr):
    code_expr2 = evaluate(state, code_expr)
    py_code = localize(state, code_expr2)

    py_code = re.sub(r"\$\w+", lambda m: _pql_PY_callback(state, m), py_code)

    try:
        res = eval(py_code)
    except Exception as e:
        raise pql_ValueError.make(state, code_expr, f"Python code provided returned an error: {e}")
    return objects.new_value_instance(res)


def pql_SQL(state: State, type_expr: ast.Expr, code_expr: ast.Expr):
    # TODO optimize for when the string is known (prefetch the variables and return Sql)
    # .. why not just compile with parameters? the types are already known
    type_ = evaluate(state, type_expr)
    code_expr2 = evaluate(state, code_expr)
    return ast.ResolveParametersString(None, type_, code_expr2)



def _canonize_default(d):
    return None if d is inspect._empty else d

def create_internal_func(fname, f):
    sig = inspect.signature(f)
    return objects.InternalFunction(fname, [
        objects.Param(None, pname, T.any, _canonize_default(sig.parameters[pname].default))
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



def pql_issubclass(state: State, expr: Object, type_expr: Object):
    "Returns whether the give object is an instance of the given type"
    inst = evaluate(state, expr)
    type_ = evaluate(state, type_expr)
    assert_type(inst.type, T.type, state, expr, 'issubclass')
    assert_type(type_.type, T.type, state, expr, 'issubclass')
    assert isinstance(inst, Type)
    assert isinstance(type_, Type)
    res = inst <= type_
    return new_value_instance(res, T.bool)

def pql_isa(state: State, expr: ast.Expr, type_expr: ast.Expr):
    "Returns whether the give object is an instance of the given type"
    inst = evaluate(state, expr)
    type_ = evaluate(state, type_expr)
    assert_type(type_.type, T.type, state, expr, 'isa')
    res = inst.isa(type_)
    return new_value_instance(res, T.bool)

def _count(state, obj: ast.Expr, table_func, name):
    obj = evaluate(state, obj)

    if obj.type <= T.table:
        code = table_func(obj.code)
    elif isinstance(obj, objects.RowInstance):
        return objects.new_value_instance(len(obj.attrs))
    else:
        if not (obj.type <= T.aggregate):
            raise pql_TypeError.make(state, None, f"Function '{name}' expected an aggregated list, but got '{obj.type}' instead. Did you forget to group?")

        obj = obj.primary_key()
        code = sql.FieldFunc(name, obj.code)

    return objects.Instance.make(code, T.int, [obj])

def pql_count(state: State, obj: ast.Expr):
    "Count how many rows are in the given table, or in the projected column."
    return _count(state, obj, sql.CountTable, 'count')


def pql_temptable(state: State, expr_ast: ast.Expr, const: objects = objects.null):
    """Generate a temporary table with the contents of the given table

    It will remain available until the db-session ends, unless manually removed.
    """
    # 'temptable' creates its own counting 'id' field. Copying existing 'id' fields will cause a collision
    # 'const table' doesn't
    expr = evaluate(state, expr_ast)
    const = localize(state, const)
    assert_type(expr.type, T.collection, state, expr_ast, 'temptable')

    # elems = dict(expr.type.elems)
    elems = expr.type.elem_dict

    if any(t <= T.unknown for t in elems.values()):
        return objects.TableInstance.make(sql.null, expr.type, [])

    name = state.unique_name("temp")    # TODO get name from table options

    if 'id' in elems and not const:
        raise pql_ValueError.make(state, None, "Field 'id' already exists. Rename it, or use 'const table' to copy it as-is.")

    table = T.table(**elems).set_options(name=name, pk=[] if const else [['id']], temporary=True)

    if not const:
        table.elems['id'] = T.t_id

    db_query(state, sql.compile_type_def(state, name, table))

    read_only, flat_columns = table_flat_for_insert(table)
    expr = exclude_fields(state, expr, set(read_only) & set(elems))
    db_query(state, sql.Insert(name, flat_columns, expr.code), expr.subqueries)

    return objects.new_table(table)

def pql_get_db_type(state: State):
    """
    Returns a string representing the type of the active database.

    Possible values are:
        - "sqlite"
        - "postgres"
    """
    assert state.access_level >= state.AccessLevels.EVALUATE
    return new_value_instance(state.db.target, T.string)



def sql_bin_op(state, op, table1, table2, name):
    t1 = evaluate(state, table1)
    t2 = evaluate(state, table2)

    if not isinstance(t1, objects.CollectionInstance):
        raise pql_TypeError.make(state, table1, f"First argument isn't a table, it's a {t1.type}")
    if not isinstance(t2, objects.CollectionInstance):
        raise pql_TypeError.make(state, table2, f"Second argument isn't a table, it's a {t2.type}")

    # TODO Smarter matching
    l1 = len(t1.type.elems)
    l2 = len(t2.type.elems)
    if l1 != l2:
        raise pql_TypeError.make(state, None, f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    code = sql.TableArith(op, [t1.code, t2.code])
    # TODO new type, so it won't look like the physical table
    return type(t1).make(code, t1.type, [t1, t2])

def pql_intersect(state: State, t1: ast.Expr, t2: ast.Expr):
    "Intersect two tables"
    return sql_bin_op(state, "INTERSECT", t1, t2, "intersect")

def pql_subtract(state: State, t1: ast.Expr, t2: ast.Expr):
    "Substract two tables (except)"
    return sql_bin_op(state, "EXCEPT", t1, t2, "subtract")

def pql_union(state: State, t1: ast.Expr, t2: ast.Expr):
    "Union two tables"
    return sql_bin_op(state, "UNION", t1, t2, "union")

def pql_concat(state: State, t1: ast.Expr, t2: ast.Expr):
    "Concatenate two tables (union all)"
    return sql_bin_op(state, "UNION ALL", t1, t2, "concatenate")




def _join(state: State, join: str, exprs: dict, joinall=False, nullable=None):

    exprs = {name: evaluate(state, value) for name,value in exprs.items()}
    for x in exprs.values():
        if not isinstance(x, objects.AbsInstance):
            raise pql_TypeError.make(state, None, f"Unexpected object type: {x}")

    if len(exprs) != 2:
        raise pql_TypeError.make(state, None, "join expected only 2 arguments")

    (a,b) = exprs.values()

    if a is objects.EmptyList or b is objects.EmptyList:
        raise pql_TypeError.make(state, None, "Cannot join on an untyped empty list")

    if isinstance(a, objects.UnknownInstance) or isinstance(b, objects.UnknownInstance):
        table_type = T.table(**{e: T.unknown for e in exprs})
        return objects.TableInstance.make(sql.unknown, table_type, [])

    if isinstance(a, objects.SelectedColumnInstance) and isinstance(b, objects.SelectedColumnInstance):
        cols = a, b
        tables = [a.parent, b.parent]
    else:
        if not ((a.type <= T.collection) and (b.type <= T.collection)):
            raise pql_TypeError.make(state, None, f"join() got unexpected values:\n * {a}\n * {b}")
        if joinall:
            tables = (a, b)
        else:
            cols = _auto_join(state, join, a, b)
            tables = [c.parent for c in cols]

    assert all((t.type <= T.collection) for t in tables)

    structs = {name: T.struct(**table.type.elem_dict) for name, table in safezip(exprs, tables)}

    # Update nullable for left/right/outer joins
    if nullable:
        structs = {name: t.replace(nullable=True) if n else t
                   for (name, t), n in safezip(structs.items(), nullable)}

    tables = [objects.alias_table_columns(t, n) for n, t in safezip(exprs, tables)]

    primary_keys = [ [name] + pk
                    for name, t in safezip(exprs, tables)
                    for pk in t.type.options.get('pk', [])
                ]
    table_type = T.table(**structs).set_options(name=state.unique_name("joinall" if joinall else "join"), pk=primary_keys)

    conds = [] if joinall else [sql.Compare('=', [sql.Name(c.type, join_names((n, c.name))) for n, c in safezip(structs, cols)])]

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
        raise pql_JoinError.make(state, None, "Cannot auto-join: No plausible relations found")
    elif auto_join_count > 1:   # Ambiguity in auto join resolution
        raise pql_JoinError.make(state, None, "Cannot auto-join: Several plausible relations found")

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
    for name, c in t1.type.elems.items():
        if (c <= T.t_relation):
            if c.elem == t2.type:
                # TODO depends on the query XXX
                yield (objects.SelectedColumnInstance(t2, T.t_id, 'id'), objects.SelectedColumnInstance(t1, c, name))

def pql_type(state: State, obj: ast.Expr):
    """
    Returns the type of the given object
    """
    inst = evaluate(state, obj)
    return inst.type

def pql_repr(state: State, obj: ast.Expr):
    """
    Returns the type of the given object
    """
    inst = evaluate(state, obj)
    try:
        return objects.new_value_instance(inst.repr(state))
    except ValueError:
        value = repr(localize(state, inst))
        return objects.new_value_instance(value)

def pql_columns(state: State, table: ast.Expr):
    """
    Returns a dictionary of {column_name: column_type}
    """
    table = evaluate(state, table)
    elems = table.type.elems
    if isinstance(elems, tuple):    # Create a tuple/list instead of dict?
        elems = {f't{i}':e for i, e in enumerate(elems)}

    return ast.Dict_(None, elems)


def pql_cast(state: State, obj: ast.Expr, type_: ast.Expr):
    "Attempt to cast an object to a specified type"
    inst = evaluate(state, obj)
    type_ = evaluate(state, type_)
    if not isinstance(type_, Type):
        raise pql_TypeError.make(state, type_, f"Cast expected a type, got {type_} instead.")

    if inst.type is type_:
        return inst

    return _cast(state, inst.type, type_, inst)

@combined_dp
def _cast(state, inst_type, target_type, inst):
    raise pql_TypeError.make(state, None, f"Cast not implemented for {inst_type}->{target_type}")

@combined_dp
def _cast(state, inst_type: T.list, target_type: T.list, inst):
    if inst is objects.EmptyList:
        return inst.replace(type=target_type)
        # return evaluate(state, ast.List_( None, target_type, []), )

    if (inst_type.elem <= target_type.elem):
        return inst

    value = inst.get_column('value')
    elem = _cast(state, value.type, target_type.elem, value)
    code = sql.Select(target_type, inst.code, [sql.ColumnAlias(elem.code, 'value')])
    return inst.replace(code=code, type=T.list[elem.type])


@combined_dp
def _cast(state, inst_type: T.aggregate, target_type: T.list, inst):
    res = _cast(state, inst_type.elem, target_type.elem, inst.elem)
    return objects.aggregate(res)   # ??

@combined_dp
def _cast(state, inst_type: T.table, target_type: T.list, inst):
    t = inst.type
    if len(t.elems) != 1:
        raise pql_TypeError.make(state, None, f"Cannot cast {inst_type} to {target_type}. Too many columns")
    if not (inst_type.elem <= target_type.elem):
        raise pql_TypeError.make(state, None, f"Cannot cast {inst_type} to {target_type}. Elements not matching")

    (elem_name, elem_type) ,= inst_type.elems.items()
    code = sql.Select(T.list[elem_type], inst.code, [sql.ColumnAlias(sql.Name(elem_type, elem_name), 'value')])

    return objects.ListInstance.make(code, T.list[elem_type], [inst])

@combined_dp
def _cast(state, inst_type: T.t_id, target_type: T.int, inst):
    return inst.replace(type=T.int)

@combined_dp
def _cast(state, inst_type: T.union[T.float, T.bool], target_type: T.int, inst):
    code = sql.Cast(T.int, "int", inst.code)
    return objects.Instance.make(code, T.int, [inst])

@combined_dp
def _cast(state, inst_type: T.union[T.int, T.bool], target_type: T.float, inst):
    code = sql.Cast(T.float, "float", inst.code)
    return objects.Instance.make(code, T.float, [inst])

@dy
def _cast(state, inst_type: T.string, target_type: T.int, inst):
    # TODO error on bad string?
    code = sql.Cast(T.int, "int", inst.code)
    return objects.Instance.make(code, T.int, [inst])

@combined_dp
def _cast(state, inst_type: T.primitive, target_type: T.string, inst):
    code = sql.Cast(T.string, "varchar", inst.code)
    return objects.Instance.make(code, T.string, [inst])

@combined_dp
def _cast(state, inst_type: T.t_relation, target_type: T.t_id, inst):
    # TODO verify same table? same type?
    return inst.replace(type=target_type)


def pql_import_table(state: State, name: ast.Expr, columns: Optional[ast.Expr] = objects.null):
    """Import an existing table from SQL

    If the columns argument is provided, only these columns will be imported.

    Example:
        >> import_table("my_sql_table", ["some_column", "another_column])
    """
    name_str = localize(state, evaluate(state, name))
    columns_whitelist = localize(state, evaluate(state, columns)) or []
    if not isinstance(columns_whitelist, list):
        raise pql_TypeError.make(state, columns, "Expected list")
    if not isinstance(name_str, str):
        raise pql_TypeError.make(state, name, "Expected string")

    columns_whitelist = set(columns_whitelist)

    # Get table type
    t = state.db.import_table_type(state, name_str, columns_whitelist)
    assert t <= T.table

    # Get table contents
    return objects.new_table(t, select_fields=bool(columns_whitelist))



def pql_connect(state: State, uri: ast.Expr):
    """
    Connect to a new database, specified by the uri
    """
    uri = localize(state, evaluate(state, uri))
    state.connect(uri)
    return objects.null

def pql_help(state: State, obj: Object = objects.null):
    """
    Provides a brief summary for a given object
    """
    if obj is objects.null:
        text = (
            "Welcome to Preql!\n\n"
            "To see the list of functions and objects available in the namespace, type 'names()'\n\n"
            "To get help for a specific function, type 'help(func_object)'\n\n"
            "For example:\n"
            "    >> help(help)\n"
        )
        return new_value_instance(text, T.string)


    lines = []
    inst = evaluate(state, obj)
    if isinstance(inst, objects.Function):
        lines = ['', inst.help_str(state),'']
        doc = inst.docstring
        if doc:
            lines += [doc]
    else:
        raise pql_TypeError.make(state, None, "help() only accepts functions at the moment")

    text = '\n'.join(lines) + '\n'
    return new_value_instance(text).replace(type=T.text)

def pql_names(state: State, obj: Object = objects.null):
    """List all names in the namespace of the given object.

    If no object is given, lists the names in the current namespace.
    """
    # TODO support all objects
    if obj is not objects.null:
        inst = evaluate(state, obj)
        if not (inst.type <= T.collection): # XXX temp.
            raise pql_TypeError.make(state, obj, "Argument to names() must be a table")
        all_vars = list(inst.all_attrs())
    else:
        all_vars = list(state.ns.get_all_vars())

    assert all(isinstance(s, str) for s in all_vars)
    names = [new_value_instance(str(s), T.string) for s in all_vars]
    return evaluate(state, ast.List_(None, T.list[T.string], names))



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


def pql_exit(state, value: Object = None):
    """Exit the current interpreter instance.

    Can be used from running code, or the REPL.

    If the current interpreter is nested within another Preql interpreter (e.g. by using debug()),
    exit() will return to the parent interpreter.
    """
    raise pql_ExitInterp(value)



def pql_import_csv(state: State, table: Object, filename: Object, header: Object = ast.Const(None, T.bool, False)):
    "Import a csv into an existing table"
    # TODO better error handling, validation
    table = evaluate(state, table)
    filename = localize(state, evaluate(state, filename))
    header = localize(state, evaluate(state, header))
    print(f"Importing CSV file: '{filename}'")

    ROWS_PER_QUERY = 1024

    cons = TableConstructor.make(table.type)
    keys = []
    rows = []

    def insert_values():
        q = sql.InsertConsts2(table.type.options['name'], keys, rows)
        db_query(state, q)


    with open(filename, 'r') as f:
        line_count = len(list(f))

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(tqdm(reader, total=line_count)):
            if i == 0:
                matched = cons.match_params(state, row)
                keys = [p.name for (p, _) in matched]

            if header and i == 0:
                # Skip first line if header=True
                continue

            values = ["'%s'" % (v.replace("'", "''")) for v in row]
            rows.append(values)

            if (i+1) % ROWS_PER_QUERY == 0:
                insert_values()
                rows = []

    if keys and rows:
        insert_values()

    return table



internal_funcs = create_internal_funcs({
    'exit': pql_exit,
    'help': pql_help,
    'names': pql_names,
    'dir': pql_names,
    'connect': pql_connect,
    'import_table': pql_import_table,
    'count': pql_count,
    'temptable': pql_temptable,
    'concat': pql_concat,
    'intersect': pql_intersect,
    'union': pql_union,
    'subtract': pql_subtract,
    'SQL': pql_SQL,
    'PY': pql_PY,
    'isa': pql_isa,
    'issubclass': pql_issubclass,
    'type': pql_type,
    'repr': pql_repr,
    'debug': pql_debug,
    '_breakpoint': pql_breakpoint,
    'get_db_type': pql_get_db_type,
    'cast': pql_cast,
    'columns': pql_columns,
    'import_csv': pql_import_csv,
})

joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param(None, 'tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param(None, 'tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param(None, 'tables')),
}
