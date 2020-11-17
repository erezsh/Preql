import inspect
from preql.compiler import cast_to_instance
import re
import csv
import inspect
from typing import Optional

from tqdm import tqdm

from .utils import safezip, listgen, re_split
from .exceptions import Signal, ExitInterp

from . import pql_objects as objects
from . import pql_ast as ast
from . import sql

from .interp_common import State, new_value_instance, assert_type
from .evaluate import evaluate, cast_to_python, db_query, TableConstructor, new_table_from_expr
from .pql_types import T, Type, union_types, Id, common_type
from .types_impl import join_names
from .casts import cast

def new_str(x):
    return new_value_instance(str(x), T.string)
def new_int(x):
    return new_value_instance(int(x), T.int)

def _pql_PY_callback(state: State, var: str):
    var = var.group()
    assert var[0] == '$'
    var_name = var[1:]
    obj = state.get_var(var_name)
    inst = evaluate(state, obj)

    if not isinstance(inst, objects.ValueInstance):
        raise Signal.make(T.TypeError, state, None, f"Cannot convert {inst} to a Python value")

    return str(inst.local_value)


def pql_PY(state: State, code_expr: T.string, code_setup: T.string.as_nullable() = objects.null):
    py_code = cast_to_python(state, code_expr)
    py_setup = cast_to_python(state, code_setup)

    py_code = re.sub(r"\$\w+", lambda m: _pql_PY_callback(state, m), py_code)

    if py_setup:
        try:
            res = exec(py_setup)
        except Exception as e:
            raise Signal.make(T.EvalError, state, code_expr, f"Python code provided returned an error: {e}")

    try:
        res = eval(py_code)
    except Exception as e:
        raise Signal.make(T.EvalError, state, code_expr, f"Python code provided returned an error: {e}")

    return objects.from_python(res)
    # return new_value_instance(res)


def pql_SQL(state: State, result_type: T.union[T.collection, T.type], sql_code: T.string):
    # TODO optimize for when the string is known (prefetch the variables and return Sql)
    # .. why not just compile with parameters? the types are already known
    return ast.ParameterizedSqlCode(result_type, sql_code)

def pql_force_eval(state: State, expr: T.object):
    "Force evaluation of expression. Execute any db queries necessary."
    return objects.new_value_instance( cast_to_python(state, expr) )

def pql_fmt(state: State, s: T.string):
    "Format given string using interpolation on variables marked as `$var`"
    _s = cast_to_python(state, s)

    tokens = re_split(r"\$\w+", _s)
    string_parts = []
    for m, t in tokens:
        if m:
            assert t[0] == '$'
            obj = state.get_var(t[1:])
            inst = cast_to_instance(state, obj)
            as_str = cast(state, inst, T.string)
            string_parts.append(as_str)
        elif t:
            string_parts.append(objects.new_value_instance(t))

    if not string_parts:
        return objects.new_value_instance('')
    elif len(string_parts) == 1:
        return string_parts[0]

    a = string_parts[0]
    for b in string_parts[1:]:
        a = ast.Arith("+", [a,b])

    return cast_to_instance(state, a)



def _canonize_default(d):
    return None if d is inspect._empty else d

def create_internal_func(fname, f):
    sig = inspect.signature(f)
    return objects.InternalFunction(fname, [
        objects.Param(pname, type_ if isinstance(type_, Type) else T.any, _canonize_default(sig.parameters[pname].default))
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
    py_api = state._py_api

    with state.use_scope(breakpoint_funcs):
        py_api.start_repl('debug> ')
    return objects.null



def pql_issubclass(state: State, inst: T.any, type: T.type):
    "Returns whether the give object is an instance of the given type"
    type_ = type
    assert_type(inst.type, T.type, state, inst, 'issubclass')
    assert_type(type_.type, T.type, state, inst, 'issubclass')
    assert isinstance(inst, Type)
    assert isinstance(type_, Type)
    res = inst <= type_
    return new_value_instance(res, T.bool)

def pql_isa(state: State, inst: T.any, type: T.type):
    "Returns whether the give object is an instance of the given type"
    type_ = type
    assert_type(type_.type, T.type, state, inst, 'isa')
    res = inst.isa(type_)
    return new_value_instance(res, T.bool)

def _count(state, obj, table_func, name):
    if obj is objects.null:
        code = sql.FieldFunc(name, sql.AllFields(T.any))
    elif obj.type <= T.table:
        code = table_func(obj.code)
    elif isinstance(obj, objects.StructInstance):
        return new_value_instance(len(obj.attrs))
    else:
        if not (obj.type <= T.aggregate):
            raise Signal.make(T.TypeError, state, None, f"Function '{name}' expected an aggregated list, but got '{obj.type}' instead. Did you forget to group?")

        obj = obj.primary_key()
        code = sql.FieldFunc(name, obj.code)

    return objects.Instance.make(code, T.int, [obj])

def pql_count(state: State, obj: T.container.as_nullable() = objects.null):
    "Count how many rows are in the given table, or in the projected column."
    return _count(state, obj, sql.CountTable, 'count')


def pql_temptable(state: State, expr: T.collection, const: T.bool.as_nullable() = objects.null):
    """Generate a temporary table with the contents of the given table

    It will remain available until the db-session ends, unless manually removed.
    """
    # 'temptable' creates its own counting 'id' field. Copying existing 'id' fields will cause a collision
    # 'const table' doesn't
    const = cast_to_python(state, const)
    assert_type(expr.type, T.collection, state, expr, 'temptable')

    name = state.unique_name("temp")    # TODO get name from table options

    return new_table_from_expr(state, name, expr, const, True)

# def pql_bfs(state: State, edges: T.collection, initial: T.collection):
#     edges = cast_to_instance(state, edges)
#     initial = cast_to_instance(state, initial)
#     breakpoint()

    # from = edges.columns["src"]
    # to = edges.columns["dst"]

    # t = TableType(get_alias(state, "bfs"))
    # add_column(t, ColumnType("edge", PrimitiveType(Int)))
    # results = instanciate_table(state, t, sql"<<dummy>>", [edges, initial])  # XXX need a better method for this
    # idx_inst = results.columns["edge"]

    # with_sql = Sql("SELECT * FROM ($(initial.code.text)) UNION SELECT e.$(to.code.text) FROM ($(edges.code.text)) e JOIN bfs ON e.$(from.code.text) = bfs.node")
    # select_sql = Sql("SELECT bfs.node AS $(idx_inst.code.text) FROM bfs")

    # results = alias_instance(select_sql, results)

    # push!(results.with_queries, ("bfs(node)", Instance(with_sql, PrimitiveType(Int), [initial, edges]))) # XXX slightly hacky
    # add_with!(state, results)


def pql_get_db_type(state: State):
    """
    Returns a string representing the type of the active database.

    Possible values are:
        - "sqlite"
        - "postgres"
    """
    assert state.access_level >= state.AccessLevels.EVALUATE
    return new_value_instance(state.db.target, T.string)



def sql_bin_op(state, op, t1, t2, name, additive=False):

    if not isinstance(t1, objects.CollectionInstance):
        raise Signal.make(T.TypeError, state, t1, f"First argument isn't a table, it's a {t1.type}")
    if not isinstance(t2, objects.CollectionInstance):
        raise Signal.make(T.TypeError, state, t2, f"Second argument isn't a table, it's a {t2.type}")

    # TODO Smarter matching?
    l1 = len(t1.type.elems)
    l2 = len(t2.type.elems)
    if l1 != l2:
        raise Signal.make(T.TypeError, state, None, f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    for e1, e2 in zip(t1.type.elems.values(), t2.type.elems.values()):
        if not (e2 <= e1):
            raise Signal.make(T.TypeError, state, None, f"Cannot {name}. Column types don't match: '{e1}' and '{e2}'")

    code = sql.TableArith(op, [t1.code, t2.code])

    return type(t1).make(code, t1.type, [t1, t2])

def pql_table_intersect(state: State, t1: T.collection, t2: T.collection):
    "Intersect two tables. Used for `t1 & t2`"
    return sql_bin_op(state, "INTERSECT", t1, t2, "intersect")

def pql_table_substract(state: State, t1: T.collection, t2: T.collection):
    "Substract two tables (except). Used for `t1 - t2`"
    if state.db.target is sql.mysql:
        raise Signal.make(T.NotImplementedError, state, t1, "MySQL doesn't support EXCEPT (yeah, really!)")
    return sql_bin_op(state, "EXCEPT", t1, t2, "subtract")

def pql_table_union(state: State, t1: T.collection, t2: T.collection):
    "Union two tables. Used for `t1 | t2`"
    return sql_bin_op(state, "UNION", t1, t2, "union", True)

def pql_table_concat(state: State, t1: T.collection, t2: T.collection):
    "Concatenate two tables (union all). Used for `t1 + t2`"
    return sql_bin_op(state, "UNION ALL", t1, t2, "concatenate", True)




def _join(state: State, join: str, exprs: dict, joinall=False, nullable=None):

    exprs = {name: evaluate(state, value) for name, value in exprs.items()}
    for x in exprs.values():
        if not isinstance(x, objects.AbsInstance):
            raise Signal.make(T.TypeError, state, None, f"Unexpected object type: {x}")

    if len(exprs) != 2:
        raise Signal.make(T.TypeError, state, None, "join expected only 2 arguments")

    (a,b) = exprs.values()

    if a is objects.EmptyList or b is objects.EmptyList:
        raise Signal.make(T.TypeError, state, None, "Cannot join on an untyped empty list")

    if isinstance(a, objects.UnknownInstance) or isinstance(b, objects.UnknownInstance):
        table_type = T.table({e: T.unknown for e in exprs})
        return objects.TableInstance.make(sql.unknown, table_type, [])

    if isinstance(a, objects.SelectedColumnInstance) and isinstance(b, objects.SelectedColumnInstance):
        cols = a, b
        tables = [a.parent, b.parent]
    else:
        if not ((a.type <= T.collection) and (b.type <= T.collection)):
            raise Signal.make(T.TypeError, state, None, f"join() got unexpected values:\n * {a}\n * {b}")
        if joinall:
            tables = (a, b)
        else:
            cols = _auto_join(state, join, a, b)
            tables = [c.parent for c in cols]

    assert all((t.type <= T.collection) for t in tables)

    structs = {name: T.struct(table.type.elems) for name, table in safezip(exprs, tables)}

    # Update nullable for left/right/outer joins
    if nullable:
        structs = {name: t.as_nullable() if n else t
                   for (name, t), n in safezip(structs.items(), nullable)}

    tables = [objects.alias_table_columns(t, n) for n, t in safezip(exprs, tables)]

    primary_keys = [ [name] + pk
                    for name, t in safezip(exprs, tables)
                    for pk in t.type.options.get('pk', [])
                ]
    table_type = T.table(structs, name=Id(state.unique_name("joinall" if joinall else "join")), pk=primary_keys)

    conds = [] if joinall else [sql.Compare('=', [sql.Name(c.type, join_names((n, c.name))) for n, c in safezip(structs, cols)])]

    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    return objects.TableInstance.make(code, table_type, [a,b])

def pql_join(state, tables):
    "Inner join two tables into a new projection {t1, t2}"
    return _join(state, "JOIN", tables)
def pql_leftjoin(state, tables):
    "Left join two tables into a new projection {t1, t2}"
    return _join(state, "LEFT JOIN", tables, nullable=[False, True])
def pql_outerjoin(state, tables):
    "Outer join two tables into a new projection {t1, t2}"
    return _join(state, "FULL OUTER JOIN", tables, nullable=[False, True])
def pql_joinall(state: State, tables):
    "Cartesian product of two tables into a new projection {t1, t2}"
    return _join(state, "JOIN", tables, True)

def _auto_join(state, join, ta, tb):
    refs1 = _find_table_reference(ta, tb)
    refs2 = _find_table_reference(tb, ta)
    auto_join_count = len(refs1) + len(refs2)
    if auto_join_count < 1:
        raise Signal.make(T.JoinError, state, None, "Cannot auto-join: No plausible relations found")
    elif auto_join_count > 1:   # Ambiguity in auto join resolution
        raise Signal.make(T.JoinError, state, None, "Cannot auto-join: Several plausible relations found")

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

def pql_type(state: State, obj: T.any):
    """
    Returns the type of the given object
    """
    return obj.type

def pql_repr(state: State, obj: T.any):
    """
    Returns the type of the given object
    """
    try:
        return new_value_instance(obj.repr(state))
    except ValueError:
        value = repr(cast_to_python(state, obj))
        return new_value_instance(value)

def pql_columns(state: State, table: T.collection):
    """
    Returns a dictionary of {column_name: column_type}
    """
    elems = table.type.elems
    if isinstance(elems, tuple):    # Create a tuple/list instead of dict?
        elems = {f't{i}':e for i, e in enumerate(elems)}

    return ast.Dict_(elems)


def pql_cast(state: State, inst: T.any, type: T.type):
    "Attempt to cast an object to a specified type"
    type_ = type
    if not isinstance(type_, Type):
        raise Signal.make(T.TypeError, state, type_, f"Cast expected a type, got {type_} instead.")

    if inst.type is type_:
        return inst

    return cast(state, inst, type_)


def pql_import_table(state: State, name: T.string, columns: T.list[T.string] = objects.null):
    """Import an existing table from SQL

    If the columns argument is provided, only these columns will be imported.

    Example:
        >> import_table("my_sql_table", ["some_column", "another_column])
    """
    name_str = cast_to_python(state, name)
    columns_whitelist = cast_to_python(state, columns) or []
    if not isinstance(columns_whitelist, list):
        raise Signal.make(T.TypeError, state, columns, "Expected list")
    if not isinstance(name_str, str):
        raise Signal.make(T.TypeError, state, name, "Expected string")

    columns_whitelist = set(columns_whitelist)

    # Get table type
    t = state.db.import_table_type(state, name_str, columns_whitelist)
    assert t <= T.table

    # Get table contents
    return objects.new_table(t, select_fields=bool(columns_whitelist))



def pql_connect(state: State, uri: T.string):
    """
    Connect to a new database, specified by the uri
    """
    uri = cast_to_python(state, uri)
    state.connect(uri)
    return objects.null

def pql_help(state: State, inst: T.any = objects.null):
    """
    Provides a brief summary for a given object
    """
    if inst is objects.null:
        text = (
            "Welcome to Preql!\n\n"
            "To see the list of functions and objects available in the namespace, type 'names()'\n"
            "To see the next page of a table preview, type '.' and then enter\n"
            "\n"
            "To get help for a specific function, type 'help(func_object)'\n\n"
            "For example:\n"
            "    >> help(help)\n"
        )
        return new_value_instance(text, T.string).replace(type=T.text)


    lines = []
    if isinstance(inst, objects.Function):
        lines = ['', inst.help_str(state),'']
        doc = inst.docstring
        if doc:
            lines += [doc]
    else:
        raise Signal.make(T.TypeError, state, None, "help() only accepts functions at the moment")

    text = '\n'.join(lines) + '\n'
    return new_value_instance(text).replace(type=T.text)

def pql_names(state: State, obj: T.any = objects.null):
    """List all names in the namespace of the given object.

    If no object is given, lists the names in the current namespace.
    """
    # TODO support all objects
    if obj is not objects.null:
        inst = obj
        if inst.type <= T.module:
            all_vars = inst.all_attrs()
        elif inst.type <= T.collection:
            all_vars = inst.all_attrs()
        else:
            raise Signal.make(T.TypeError, state, obj, "Argument to names() must be a table or module")
    else:
        all_vars = (state.ns.get_all_vars())

    assert all(isinstance(s, str) for s in all_vars)
    tuples = [sql.Tuple(T.list[T.string], [new_str(n).code,new_str(v.type).code]) for n,v in all_vars.items()]

    table_type = T.table(dict(name=T.string, type=T.string))
    return objects.new_const_table(state, table_type, tuples)


def pql_tables(state: State):
    names = state.db.list_tables()
    values = [(name, state.db.import_table_type(state, name, None)) for name in names]
    tuples = [sql.Tuple(T.list[T.string], [new_str(n).code,new_str(t).code]) for n,t in values]

    table_type = T.table(dict(name=T.string, type=T.string))
    return objects.new_const_table(state, table_type, tuples)



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


def pql_exit(state, value: T.int.as_nullable() = None):
    """Exit the current interpreter instance.

    Can be used from running code, or the REPL.

    If the current interpreter is nested within another Preql interpreter (e.g. by using debug()),
    exit() will return to the parent interpreter.
    """
    raise ExitInterp(value)



def pql_import_csv(state: State, table: T.table, filename: T.string, header: T.bool = ast.Const(T.bool, False)):
    "Import a csv into an existing table"
    # TODO better error handling, validation
    filename = cast_to_python(state, filename)
    header = cast_to_python(state, header)
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


def _rest_func_endpoint(state, func):
    from starlette.responses import JSONResponse
    async def callback(request):
        params = [objects.new_value_instance(v) for k, v in request.path_params.items()]
        expr = ast.FuncCall(func, params)
        res = evaluate(state, expr)
        res = cast_to_python(state, res)
        return JSONResponse(res)
    return callback

def _rest_table_endpoint(state, table):
    from starlette.responses import JSONResponse
    async def callback(request):
        tbl = table
        params = dict(request.query_params)
        if params:
            conds = [ast.Compare('=', [ast.Name(k), objects.new_value_instance(v)])
                     for k, v in params.items()]
            expr = ast.Selection(tbl, conds)
            tbl = evaluate(state, expr)
        res = cast_to_python(state, tbl)
        return JSONResponse(res)
    return callback


def pql_serve_rest(state: State, endpoints: T.struct, port: T.int = new_int(8080)):

    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError:
        raise Signal.make(T.ImportError, state, None, "starlette not installed! Run 'pip install starlette'")

    try:
        import uvicorn
    except ImportError:
        raise Signal.make(T.ImportError, state, None, "uvicorn not installed! Run 'pip install uvicorn'")

    port_ = cast_to_python(state, port)

    async def root(request):
        return JSONResponse(list(endpoints.attrs))

    routes = [
        Route("/", endpoint=root)
    ]

    for func_name, func in endpoints.attrs.items():
        path = "/" + func_name
        if func.type <= T.function:
            for p in func.params:
                path += "/{%s}" % p.name

            routes.append(Route(path, endpoint=_rest_func_endpoint(state, func)))
        elif func.type <= T.collection:
            routes.append(Route(path, endpoint=_rest_table_endpoint(state, func)))
        else:
            raise Signal.make(T.TypeError, state, func, f"Expected a function or a table, got {func.type}")

    app = Starlette(debug=True, routes=routes)

    uvicorn.run(app, port=port_)
    return objects.null


internal_funcs = create_internal_funcs({
    'exit': pql_exit,
    'help': pql_help,
    'names': pql_names,
    'tables': pql_tables,
    'dir': pql_names,
    'connect': pql_connect,
    'import_table': pql_import_table,
    'count': pql_count,
    'temptable': pql_temptable,
    'table_concat': pql_table_concat,
    'table_intersect': pql_table_intersect,
    'table_union': pql_table_union,
    'table_subtract': pql_table_substract,
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
    'serve_rest': pql_serve_rest,
    'force_eval': pql_force_eval,
    'fmt': pql_fmt,
})

joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param('tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param('tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param('tables')),
    'outerjoin': objects.InternalFunction('outerjoin', [], pql_outerjoin, objects.Param('tables')),
}
