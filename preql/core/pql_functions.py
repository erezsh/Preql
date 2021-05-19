import inspect
import re
import os
from datetime import datetime
import csv
import itertools

import rich.progress
import runtype

from preql.utils import safezip, listgen, re_split
from preql.docstring.autodoc import autodoc, AutoDocError
from preql.context import context

from .exceptions import Signal, ExitInterp
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import pyvalue_inst, assert_type, cast_to_python_string, cast_to_python_int, cast_to_python
from .state import get_var, get_db, use_scope, unique_name, get_db_target, require_access, AccessLevels
from .evaluate import evaluate, db_query, TableConstructor, new_table_from_expr, new_table_from_rows
from .pql_types import T, Type, Id
from .types_impl import join_names
from .casts import cast
from .compiler import cast_to_instance

def new_str(x):
    return pyvalue_inst(str(x), T.string)

def _pql_PY_callback(var: str):
    var = var.group()
    assert var[0] == '$'
    var_name = var[1:]
    obj = get_var(var_name)
    inst = evaluate( obj)

    if not isinstance(inst, objects.ValueInstance):
        raise Signal.make(T.TypeError, None, f"Cannot convert {inst} to a Python value")

    return str(inst.local_value)


def pql_PY(code_expr: T.string, code_setup: T.string.as_nullable() = objects.null):
    """Evaluate the given Python expression and convert the result to a Preql object

    Parameters:
        code_expr: The Python expression to evaluate
        code_setup: Setup code to prepare for the evaluation

    Note:
        This function is still experemental, and should be used with caution.

    Example:
        >> PY("sys.version", "import sys")
        "3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10)"
    """
    py_code = cast_to_python_string(code_expr)
    py_setup = cast_to_python_string(code_setup)

    py_code = re.sub(r"\$\w+", _pql_PY_callback, py_code)

    if py_setup:
        try:
            res = exec(py_setup)
        except Exception as e:
            raise Signal.make(T.EvalError, code_expr, f"Python code provided returned an error: {e}")

    try:
        res = eval(py_code)
    except Exception as e:
        raise Signal.make(T.EvalError, code_expr, f"Python code provided returned an error: {e}")

    return objects.from_python(res)
    # return pyvalue_inst(res)

def pql_inspect_sql(obj: T.object):
    """Returns the SQL code that would be executed to evaluate the given object

    """
    if not isinstance(obj, objects.Instance):
        raise Signal.make(T.TypeError, None, f"inspect_sql() expects a concrete object. Instead got: {obj.type}")
    s = get_db().compile_sql(obj.code, obj.subqueries)
    return objects.ValueInstance.make(sql.make_value(s), T.text, [], s)


def pql_SQL(result_type: T.union[T.table, T.type], sql_code: T.string):
    """Create an object with the given SQL evaluation code, and given result type.

    The object will only be evaluated when required by the program flow.

    Using $var_name in the code will embed it in the query. Both primitives and tables are supported.

    A special `$self` variable allows to perform recursion, if supported by the dialect.


    Parameters:
        result_type: The expected type of the result of the SQL query
        sql_code: The SQL code to be evaluated

    Example:
        >> ["a", "b"]{item: SQL(string, "$item || '!'")}
        table  =2
        ┏━━━━━━━┓
        ┃ item  ┃
        ┡━━━━━━━┩
        │ a!    │
        │ b!    │
        └───────┘

        >> x = ["a", "b", "c"]
        >> SQL(type(x), "SELECT item || '!' FROM $x")
        table  =3
        ┏━━━━━━━┓
        ┃ item  ┃
        ┡━━━━━━━┩
        │ a!    │
        │ b!    │
        │ c!    │
        └───────┘
    """
    # TODO optimize for when the string is known (prefetch the variables and return Sql)
    # .. why not just compile with parameters? the types are already known
    return ast.ParameterizedSqlCode(result_type, sql_code)

def pql_force_eval(expr: T.object):
    """Forces the evaluation of the given expression.

    Executes any db queries necessary.
    """
    return objects.pyvalue_inst( cast_to_python(expr) )

def pql_fmt(s: T.string):
    """Format the given string using interpolation on variables marked as `$var`

    Example:
        >> ["a", "b", "c"]{item: fmt("$item!")}
        table  =3
        ┏━━━━━━━┓
        ┃ item  ┃
        ┡━━━━━━━┩
        │ a!    │
        │ b!    │
        │ c!    │
        └───────┘
    """
    _s = cast_to_python_string(s)

    tokens = re_split(r"\$\w+", _s)
    string_parts = []
    for m, t in tokens:
        if m:
            assert t[0] == '$'
            obj = get_var(t[1:])
            inst = cast_to_instance(obj)
            as_str = cast(inst, T.string)
            string_parts.append(as_str)
        elif t:
            string_parts.append(objects.pyvalue_inst(t))

    if not string_parts:
        return objects.pyvalue_inst('')
    elif len(string_parts) == 1:
        return string_parts[0]

    a = string_parts[0]
    for b in string_parts[1:]:
        a = ast.BinOp("+", [a,b])

    return cast_to_instance(a)



def _canonize_default(d):
    return None if d is inspect._empty else d

def create_internal_func(fname, f):
    sig = inspect.signature(f)
    return objects.InternalFunction(fname, [
        objects.Param(pname, type_ if isinstance(type_, Type) else T.any,
                      _canonize_default(sig.parameters[pname].default))
        for pname, type_ in list(f.__annotations__.items())
    ], f)

def create_internal_funcs(d):
    new_d = {}
    for names, f in d.items():
        if isinstance(names, str):
            names = (names,)
        for name in names:
            new_d[name] = create_internal_func(name, f)
    return new_d

def create_internal_properties(d):
    return {k: objects.Property(v) for k, v in create_internal_funcs(d).items()}


def pql_brk_continue():
    "Continue the execution of the code (exit debug interpreter)"
    pql_exit(objects.null)


def pql_breakpoint():
    """Creates a Python breakpoint.

    For a Preql breakpoint, use debug() instead
    """
    breakpoint()
    return objects.null

def pql_debug():
    """Breaks the execution of the interpreter, and enters into a debug
    session using the REPL environment.

    Use `c()` to continue the execution.
    """
    state = context.state
    py_api = state.interp._py_api

    with use_scope(breakpoint_funcs):
        py_api.start_repl('debug> ')
    return objects.null



def pql_issubclass(a: T.type, b: T.type):
    """Checks if type 'a' is a subclass of type 'b'

    Examples:
        >> issubclass(int, number)
        true
        >> issubclass(int, table)
        false
        >> issubclass(list, table)
        true
    """
    assert_type(a.type, T.type, a, 'issubclass')
    assert_type(b.type, T.type, b, 'issubclass')
    assert isinstance(a, Type)
    assert isinstance(b, Type)
    return pyvalue_inst(a <= b, T.bool)

def pql_isa(obj: T.any, type: T.type):
    """Checks if the give object is an instance of the given type

    Examples:
        >> isa(1, int)
        true
        >> isa(1, string)
        false
        >> isa(1.2, number)
        true
        >> isa([1], table)
        true
    """
    assert_type(type.type, T.type, obj, 'isa')
    res = obj.isa(type)
    return pyvalue_inst(res, T.bool)

def _count(obj, table_func, name='count'):
    if obj is objects.null:
        code = sql.FieldFunc(name, sql.AllFields(T.any))
    elif obj.type <= T.table:
        code = table_func(obj.code)
    elif isinstance(obj, objects.StructInstance) and not obj.type <= T.aggregated:
        # XXX is count() even the right method for this?
        return pyvalue_inst(len(obj.attrs))

    elif obj.type <= T.projected[T.json_array]:
        code = sql.JsonLength(obj.code)
    else:
        if not obj.type <= T.aggregated:
            msg = f"Function '{name}' expected an aggregated list, but got '{obj.type}' instead. Did you forget to group?"
            raise Signal.make(T.TypeError, None, msg)

        obj = obj.primary_key()
        code = sql.FieldFunc('count', obj.code)

    return objects.Instance.make(code, T.int, [obj])

def pql_count(obj: T.container.as_nullable() = objects.null):
    """Count how many rows are in the given table, or in the projected column.

    If no argument is given, count all the rows in the current projection.

    Examples:
        >> count([0..10])
        10
        >> [0..10]{ => count() }
        table  =1
        ┏━━━━━━━┓
        ┃ count ┃
        ┡━━━━━━━┩
        │    10 │
        └───────┘
        >> [0..10]{ => count(item) }
        table  =1
        ┏━━━━━━━┓
        ┃ count ┃
        ┡━━━━━━━┩
        │    10 │
        └───────┘
    """
    return _count(obj, sql.CountTable)


def pql_temptable(expr: T.table, const: T.bool.as_nullable() = objects.null):
    """Generate a temporary table with the contents of the given table

    It will remain available until the database session ends, unless manually removed.

    Parameters:
        expr: the table expression to create the table from
        const: whether the resulting table may be changed or not.

    Note:
        A non-const table creates its own `id` field.
        Trying to copy an existing id field into it will cause a collision
    """
    # 'temptable' creates its own counting 'id' field. Copying existing 'id' fields will cause a collision
    # 'const table' doesn't
    const = cast_to_python(const)
    assert_type(expr.type, T.table, expr, 'temptable')

    name = unique_name("temp")    # TODO get name from table options

    return new_table_from_expr(name, expr, const, True)


def pql_get_db_type():
    """Returns a string representing the type of the active database.

    Example:
        >> get_db_type()
        "sqlite"
    """
    require_access(AccessLevels.EVALUATE)
    return pyvalue_inst(get_db_target(), T.string)



def sql_bin_op(op, t1, t2, name):

    if not isinstance(t1, objects.CollectionInstance):
        raise Signal.make(T.TypeError, t1, f"First argument isn't a table, it's a {t1.type}")
    if not isinstance(t2, objects.CollectionInstance):
        raise Signal.make(T.TypeError, t2, f"Second argument isn't a table, it's a {t2.type}")

    # TODO Smarter matching?
    l1 = len(t1.type.elems)
    l2 = len(t2.type.elems)
    if l1 != l2:
        msg = f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)"
        raise Signal.make(T.TypeError, None, msg)

    for e1, e2 in zip(t1.type.elems.values(), t2.type.elems.values()):
        if not e2 <= e1:
            raise Signal.make(T.TypeError, None, f"Cannot {name}. Column types don't match: '{e1}' and '{e2}'")

    code = sql.TableArith(op, [t1.code, t2.code])

    return type(t1).make(code, t1.type, [t1, t2])

def pql_table_intersect(t1: T.table, t2: T.table):
    "Intersect two tables. Used for `t1 & t2`"
    return sql_bin_op("INTERSECT", t1, t2, "intersect")

def pql_table_substract(t1: T.table, t2: T.table):
    "Substract two tables (except). Used for `t1 - t2`"
    if get_db_target() is sql.mysql:
        raise Signal.make(T.NotImplementedError, t1, "MySQL doesn't support EXCEPT (yeah, really!)")
    return sql_bin_op("EXCEPT", t1, t2, "subtract")

def pql_table_union(t1: T.table, t2: T.table):
    "Union two tables. Used for `t1 | t2`"
    return sql_bin_op("UNION", t1, t2, "union")

def pql_table_concat(t1: T.table, t2: T.table):
    "Concatenate two tables (union all). Used for `t1 + t2`"
    if isinstance(t1, objects.EmptyListInstance):
        return t2
    if isinstance(t2, objects.EmptyListInstance):
        return t1
    return sql_bin_op("UNION ALL", t1, t2, "concatenate")


def _get_table(t):
    if isinstance(t, objects.SelectedColumnInstance):
        return t.parent

    if not isinstance(t, objects.CollectionInstance):
        raise Signal.make(T.TypeError, None, f"join() arguments must be tables")
    return t

def _join2(a, b):
    if isinstance(a, objects.SelectedColumnInstance) and isinstance(b, objects.SelectedColumnInstance):
        return [a, b]

    if not ((a.type <= T.table) and (b.type <= T.table)):
        a = a.type.repr()
        b = b.type.repr()
        raise Signal.make(T.TypeError, None, f"join() arguments must be of same type. Instead got:\n * {a}\n * {b}")

    return _auto_join(a, b)

def _auto_join(ta, tb):
    refs1 = _find_table_reference(ta, tb)
    refs2 = _find_table_reference(tb, ta)
    auto_join_count = len(refs1) + len(refs2)
    if auto_join_count < 1:
        raise NoAutoJoinFound(ta, tb)
    elif auto_join_count > 1:   # Ambiguity in auto join resolution
        raise Signal.make(T.JoinError, None, "Cannot auto-join: Several plausible relations found")

    if len(refs1) == 1:
        dst, src = refs1[0]
    elif len(refs2) == 1:
        src, dst = refs2[0]
    else:
        assert False

    return src, dst


def _join(join: str, on, exprs_dict: dict, joinall=False, nullable=None):
    names = list(exprs_dict)
    exprs = [evaluate( value) for value in exprs_dict.values()]

    if len(exprs_dict.attrs) < 2:
        raise Signal.make(T.ValueError, None, f"Need at least two tables to join. Got {len(exprs_dict.attrs)}")


    # Validation and edge cases
    for x in exprs:
        if not isinstance(x, objects.AbsInstance):
            raise Signal.make(T.TypeError, None, f"Unexpected object type: {x}")

    for e in exprs:
        if e is objects.EmptyList:
            raise Signal.make(T.TypeError, None, "Cannot join on an untyped empty list")

        if isinstance(e, objects.UnknownInstance):
            table_type = T.table({n: T.unknown for n in names})
            return objects.TableInstance.make(sql.unknown, table_type, [])

    # Initialization
    tables = [_get_table(x) for x in exprs]
    assert all((t.type <= T.table) for t in tables)

    structs = {name: T.struct(table.type.elems) for name, table in safezip(names, tables)}

    if nullable:
        # Update nullable for left/right/outer joins
        structs = {name: t.as_nullable() if n else t
                   for (name, t), n in safezip(structs.items(), nullable)}

    tables = [objects.alias_table_columns(t, n) for n, t in safezip(names, tables)]

    primary_keys = [[name] + pk
                    for name, t in safezip(names, tables)
                    for pk in t.type.options.get('pk', [])
                   ]
    table_type = T.table(structs, name=Id(unique_name("joinall" if joinall else "join")), pk=primary_keys)

    conds = []
    if joinall:
        for e in exprs:
            if not isinstance(e, objects.CollectionInstance):
                raise Signal.make(T.TypeError, None, f"joinall() expected tables. Got {e}")
    elif on is not objects.null:
        attrs = {n: objects.make_instance_from_name(table_type.elems[n], n) for n in names}
        with use_scope({n: objects.projected(c) for n, c in attrs.items()}):
            on = cast_to_instance(on)
        conds.append(on.code)
    else:
        if len(exprs) < 2:
            raise Signal.make(T.TypeError, None, "join expected at least 2 arguments")

        joined_exprs = set()
        for (na, ta), (nb, tb) in itertools.combinations(safezip(names, exprs), 2):
            try:
                cols = _join2(ta, tb)
                cond = sql.Compare('=', [sql.Name(c.type, join_names((n, c.name))) for n, c in safezip([na, nb], cols)])
                conds.append(cond)
                joined_exprs |= {id(ta), id(tb)}
            except NoAutoJoinFound as e:
                pass

        if {id(e) for e in exprs} != set(joined_exprs):
            # TODO better error!!! table name?? specific failed auto-join?
            s = ', '.join(repr(t.type) for t in exprs)
            raise Signal.make(T.JoinError, None, f"Cannot auto-join: No plausible relations found between {s}")


    code = sql.Join(table_type, join, [t.code for t in tables], conds)
    return objects.TableInstance.make(code, table_type, exprs)


def pql_join(on, tables):
    """Inner-join any number of tables.

    Each argument is expected to be one of -
    (1) A column to join on. Columns are attached to specific tables. or
    (2) A table to join on. The column will be chosen automatically, if there is no ambiguity.
    Connections are made according to the relationships in the declaration of the table.

    Parameters:
        $on: Optional special keyword argument for specifying join condition.
             When specified, auto-join will be skipped.
        tables: Provided as keyword arguments, in the form of <name>: <table>.
                Each keyword argument must be either a column, or a table.

    Returns:
        A new table, where each column is a struct representing one of
        the joined tables.

    Examples:
        >> join(a: [0].item, b: [0].item)
        table join46 =1
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ a           ┃ b           ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ {'item': 0} │ {'item': 0} │
        └─────────────┴─────────────┘

        >> join(a: [1..5].item, b: [3..8].item) {...a}
        table  =2
        ┏━━━━━━━┓
        ┃  item ┃
        ┡━━━━━━━┩
        │     3 │
        │     4 │
        └───────┘

         >> leftjoin(a: [1,3], b: [1,2], $on: a.item > b.item)
        table join78 =3
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ a           ┃ b              ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ {'item': 1} │ {'item': None} │
        │ {'item': 3} │ {'item': 1}    │
        │ {'item': 3} │ {'item': 2}    │
        └─────────────┴────────────────┘

        >> join(c: Country, l: Language) {...c, language: l.name}
    """
    return _join("JOIN", on, tables)

def pql_leftjoin(on, tables):
    """Left-join any number of tables

    See `join`
    """
    return _join("LEFT JOIN", on, tables, nullable=[False, True])

def pql_outerjoin(on, tables):
    """Outer-join any number of tables

    See `join`
    """
    return _join("FULL OUTER JOIN", on, tables, nullable=[False, True])

def pql_joinall(on, tables):
    """Cartesian product of any number of tables

    See `join`

    Example:
        >> joinall(a: [0,1], b: ["a", "b"])
            table joinall14 =4
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
        ┃ a           ┃ b             ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
        │ {'item': 0} │ {'item': 'a'} │
        │ {'item': 0} │ {'item': 'b'} │
        │ {'item': 1} │ {'item': 'a'} │
        │ {'item': 1} │ {'item': 'b'} │
        └─────────────┴───────────────┘
    """
    return _join("JOIN", on, tables, True)

class NoAutoJoinFound(Exception):
    pass


@listgen
def _find_table_reference(t1, t2):
    for name, c in t1.type.elems.items():
        if c <= T.t_relation:
            rel = c.options['rel']
            if rel['table'] == t2.type:     # if same table
                yield t2.get_attr(rel['column']), objects.SelectedColumnInstance(t1, c, name)

def pql_type(obj: T.any):
    """Returns the type of the given object

    Example:
        >> type(1)
        int
        >> type([1])
        list[item: int]
        >> type(int)
        type
    """
    return obj.type


def pql_repr(obj: T.any):
    """Returns the representation text of the given object
    """
    if obj.type <= T.projected | T.aggregated:
        raise Signal.make(T.CompileError, obj, "repr() cannot run in projected/aggregated mode")

    try:
        return pyvalue_inst(obj.repr())
    except ValueError:
        value = repr(cast_to_python(obj))
        return pyvalue_inst(value)

def pql_columns(obj: T.container):
    """Returns a dictionary `{column_name: column_type}` for the given table

    Example:
        >> columns([0])
        {item: int}
    """

    elems = obj.type.elems
    if isinstance(elems, tuple):    # Create a tuple/list instead of dict?
        elems = {f't{i}':e for i, e in enumerate(elems)}

    return ast.Dict_(elems)


def pql_cast(obj: T.any, target_type: T.type):
    """Attempt to cast an object to a specified type

    The resulting object will be of type `target_type`, or a `TypeError`
    exception will be thrown.

    Parameters:
        obj: The object to cast
        target_type: The type to cast to

    """
    type_ = target_type
    if not isinstance(type_, Type):
        raise Signal.make(T.TypeError, type_, f"Cast expected a type, got {type_} instead.")

    if obj.type is type_:
        return obj

    return cast(obj, type_)


def pql_import_table(name: T.string, columns: T.list[T.string].as_nullable() = objects.null):
    """Import an existing table from the database, and fill in the types automatically.

    Parameters:
        name: The name of the table to import
        columns: If this argument is provided, only these columns will be imported.

    Example:
        >> import_table("my_sql_table", ["some_column", "another_column])
    """
    name_str = cast_to_python_string(name)
    assert isinstance(name_str, str)
    columns_whitelist = cast_to_python(columns)
    if columns_whitelist is not None:
        assert isinstance(columns_whitelist, list)
        columns_whitelist = set(columns_whitelist)

    # Get table type
    t = get_db().import_table_type(name_str, columns_whitelist)
    assert t <= T.table

    # Get table contents
    return objects.new_table(t, select_fields=bool(columns_whitelist))



def pql_connect(uri: T.string, load_all_tables: T.bool = ast.false, auto_create: T.bool = ast.false):
    """Connect to a new database, specified by the uri

    Parameters:
        uri: A string specifying which database to connect to (e.g. "sqlite:///test.db")
        load_all_tables: If true, loads all the tables in the database into the global namespace.
        auto_create: If true, creates the database if it doesn't already exist (Sqlite only)

    Example:
        >> connect("sqlite://:memory:")     // Connect to a database in memory
    """
    state = context.state
    uri = cast_to_python_string(uri)
    load_all_tables = cast_to_python(load_all_tables)
    auto_create = cast_to_python(auto_create)
    state.connect(uri, auto_create=auto_create)
    if load_all_tables:
        state.interp.load_all_tables()     # XXX
    return objects.null

def pql_help(inst: T.any = objects.null):
    """Provides a brief summary for the given object
    """
    if inst is objects.null:
        text = (
            "To see the list of functions and objects available in the namespace, type '[b]names()[/b]'\n"
            "To see the next page of a table preview, type '.' and then enter\n"
            "\n"
            "To get help for a specific function, type '[b]help(an_object)[/b]'\n\n"
            "For example:\n"
            "    >> help(help)\n"
        )
        return pyvalue_inst(text, T.string).replace(type=T._rich)


    lines = []

    table_type = None
    if inst.type <= T.table:
        table_type = inst.type
    elif isinstance(inst, Type) and inst <= T.table:
        table_type = inst

    if table_type:
        inst = T.table

    try:
        doc = autodoc(inst).print_text()    # TODO maybe html
        if doc:
            lines += [doc]
    except NotImplementedError:
        inst_repr = inst.rich_repr()
        lines = [f"No help available for {inst_repr}"]
    except runtype.DispatchError:
        type_repr = inst.type.rich_repr()
        lines += [f"<doc not available yet for object of type '{type_repr}>'"]
    except AutoDocError:
        inst_repr = inst.rich_repr()
        lines += [f"<error generating documentation for object '{inst_repr}'>"]



    text = '\n'.join(lines) + '\n'
    return pyvalue_inst(text).replace(type=T._rich)

def _get_doc(v):
    s = ''
    if v.type <= T.function and v.docstring:
        s = v.docstring.splitlines()[0]
    return new_str(s).code

def pql_names(obj: T.any = objects.null):
    """List all names in the namespace of the given object.

    If no object is given, lists the names in the current namespace.
    """
    state = context.state
    if obj is objects.null:
        all_vars = (state.ns.get_all_vars())
    else:
        all_vars = obj.all_attrs()

    assert all(isinstance(s, str) for s in all_vars)
    all_vars = list(all_vars.items())
    all_vars.sort()
    tuples = [sql.Tuple(T.list[T.string], [new_str(n).code,new_str(v.type).code, _get_doc(v)]) for n,v in all_vars]

    table_type = T.table(dict(name=T.string, type=T.string, doc=T.string))
    return objects.new_const_table(table_type, tuples)


def pql_tables():
    """Returns a table of all the persistent tables in the database.

    The resulting table has two columns: name, and type.
    """
    db = get_db()
    names = db.list_tables()
    values = [(name, db.import_table_type(name, None)) for name in names]
    tuples = [sql.Tuple(T.list[T.string], [new_str(n).code,new_str(t).code]) for n,t in values]

    table_type = T.table(dict(name=T.string, type=T.string))
    return objects.new_const_table(table_type, tuples)


def pql_env_vars():
    """Returns a table of all the environment variables.

    The resulting table has two columns: name, and value.
    """
    tuples = [sql.Tuple(T.list[T.string], [new_str(n).code,new_str(t).code]) for n,t in os.environ.items()]

    table_type = T.table({'name':T.string, 'value': T.string})
    return objects.new_const_table(table_type, tuples)


breakpoint_funcs = create_internal_funcs({
    ('c', 'continue'): pql_brk_continue
})


def pql_exit(value: T.any.as_nullable() = None):
    """Exit the current interpreter instance.

    Can be used from running code, or the REPL.

    If the current interpreter is nested within another Preql interpreter (e.g. by using debug()),
    exit() will return to the parent interpreter.

    """
    raise ExitInterp(value)



def import_pandas(dfs):
    """Import pandas.DataFrame instances into SQL tables
    """
    import pandas as pd
    def normalize_item(i):
        if pd.isna(i):
            return None
        i = i.item() if hasattr(i, 'item') else i
        return i

    for name, df in dfs.items():
        if isinstance(df, pd.Series):
            cols = ['key', 'value']
            rows = [(dt.to_pydatetime() if isinstance(dt, datetime) else dt,v) for dt, v in df.items()]
        else:
            assert isinstance(df, pd.DataFrame)
            cols = list(df)
            rows = [[normalize_item(i) for i in rec]
                    for rec in df.to_records()]
            rows = [ row[1:] for row in rows ]    # drop index

        yield new_table_from_rows(name, cols, rows)

def pql_import_json(table_name: T.string, uri: T.string):
    """Imports a json file into a new table.

    Returns the newly created table.

    Parameters:
        table_name: The name of the table to create
        uri: A path or URI to the JSON file

    Note:
        This function requires the `pandas` Python package.
    """
    table_name = cast_to_python_string(table_name)
    uri = cast_to_python_string(uri)
    print(f"Importing JSON file: '{uri}'")

    import pandas
    df = pandas.read_json(uri)
    tbl ,= import_pandas({table_name: df})
    return tbl



def pql_import_csv(table: T.table, filename: T.string, header: T.bool = ast.Const(T.bool, False)):
    """Import a csv file into an existing table

    Parameters:
        table: A table into which to add the rows.
        filename: A path to the csv file
        header: If true, skips the first line
    """
    # TODO better error handling, validation

    filename = cast_to_python_string(filename)
    header = cast_to_python(header)
    msg = f"Importing CSV file: '{filename}'"

    ROWS_PER_QUERY = 1024

    cons = TableConstructor.make(table.type)
    keys = []
    rows = []

    def insert_values():
        q = sql.InsertConsts2(table.type.options['name'], keys, rows)
        db_query(q)


    try:
        with open(filename, 'r', encoding='utf8') as f:
            line_count = len(list(f))
            f.seek(0)

            reader = csv.reader(f)
            for i, row in enumerate(rich.progress.track(reader, total=line_count, description=msg)):
                if i == 0:
                    matched = cons.match_params(row)
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
    except FileNotFoundError as e:
        raise Signal.make(T.FileError, None, str(e))


def _rest_func_endpoint(func):
    from starlette.responses import JSONResponse
    async def callback(request):
        params = [objects.pyvalue_inst(v) for k, v in request.path_params.items()]
        expr = ast.FuncCall(func, params)
        res = evaluate( expr)
        res = cast_to_python(res)
        return JSONResponse(res)
    return callback

def _rest_table_endpoint(table):
    from starlette.responses import JSONResponse
    async def callback(request):
        tbl = table
        params = dict(request.query_params)
        if params:
            conds = [ast.Compare('=', [ast.Name(k), objects.pyvalue_inst(v)])
                     for k, v in params.items()]
            expr = ast.Selection(tbl, conds)
            tbl = evaluate( expr)
        res = cast_to_python(tbl)
        return JSONResponse(res)
    return callback


def pql_serve_rest(endpoints: T.struct, port: T.int = pyvalue_inst(8080)):
    """Start a starlette server (HTTP) that exposes the current namespace as REST API

    Parameters:
        endpoints: A struct of type (string => function), mapping names to the functions.
        port: A port from which to serve the API

    Note:
        Requires the `starlette` package for Python. Run `pip install starlette`.

    Example:
        >> func index() = "Hello World!"
        >> serve_rest({index: index})
        INFO     Started server process [85728]
        INFO     Waiting for application startup.
        INFO     Application startup complete.
        INFO     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
    """

    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError:
        raise Signal.make(T.ImportError, None, "starlette not installed! Run 'pip install starlette'")

    try:
        import uvicorn
    except ImportError:
        raise Signal.make(T.ImportError, None, "uvicorn not installed! Run 'pip install uvicorn'")

    port_ = cast_to_python_int(port)

    async def root(_request):
        return JSONResponse(list(endpoints.attrs))

    routes = [
        Route("/", endpoint=root)
    ]

    for func_name, func in endpoints.attrs.items():
        path = "/" + func_name
        if func.type <= T.function:
            for p in func.params:
                path += "/{%s}" % p.name

            routes.append(Route(path, endpoint=_rest_func_endpoint(func)))
        elif func.type <= T.table:
            routes.append(Route(path, endpoint=_rest_table_endpoint(func)))
        else:
            raise Signal.make(T.TypeError, func, f"Expected a function or a table, got {func.type}")

    app = Starlette(debug=True, routes=routes)

    uvicorn.run(app, port=port_)
    return objects.null


def pql_table_add_index(table, column_name: T.string, unique: T.bool = ast.false):
    """Add an index to the table, to optimize filtering operations.

    A method of the `table` type.

    Parameters:
        column_name: The name of the column to add index
        unique: If true, every value in the column is expected to be unique 

    Note:
        Future versions of this function will accept several columns.

    Example:
        >> table x = [1,2,3]{item}
        >> x.add_index("item")
    """
    try:
        table_name = table.type.options['name']
    except KeyError:
        raise Signal.make(T.TypeError, None, f"Can only add indexes to persistent tables")

    unique = cast_to_python(unique)
    column_name = cast_to_python(column_name)
    
    index_name = unique_name(f'index_{table_name.repr_name}_{column_name}')

    code = sql.AddIndex(Id(index_name), table_name, column_name, unique=unique)
    db_query(code)

    return objects.null

T.table.proto_attrs.update(create_internal_funcs({
    'add_index': pql_table_add_index
}))


def _make_datetime_method(func_name):
    def f(datetime):
        return context.state.interp.call_func(func_name, [datetime])
    return f

_datetime_methods = 'hour', 'minute', 'day', 'month', 'year', 'day_of_week', 'week_of_year'

T.datetime.proto_attrs.update(create_internal_properties({
    n: _make_datetime_method(n) for n in _datetime_methods
}))


internal_funcs = create_internal_funcs({
    'exit': pql_exit,
    'help': pql_help,
    'names': pql_names,
    'tables': pql_tables,
    'env_vars': pql_env_vars,
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
    'inspect_sql': pql_inspect_sql,
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
    'import_json': pql_import_json,
    'serve_rest': pql_serve_rest,
    'force_eval': pql_force_eval,
    'fmt': pql_fmt,
})

_joins = {
    'join': pql_join,
    'joinall': pql_joinall,
    'leftjoin': pql_leftjoin,
    'outerjoin': pql_outerjoin,
}

joins = {
    k: objects.InternalFunction(k, [objects.Param('$on', default=objects.null)], v, objects.Param('tables'))
    for k, v in _joins.items()
}
