from typing import Optional

from .utils import safezip, listgen, SafeDict
from .exceptions import pql_TypeError, pql_JoinError

from . import pql_objects as objects
from . import pql_types as types
from . import pql_ast as ast
from . import sql

from .compiler import compile_remote, instanciate_table, compile_type_def, _make_name
from .interp_common import State, get_alias, sql_repr
from .evaluate import simplify, evaluate, localize

def _pql_SQL_callback(state: State, var: str, instances):
    var = var.group()
    assert var[0] == '$'
    var_name = var[1:]
    obj = state.get_var(var_name)

    if isinstance(obj, types.TableType):
        # This branch isn't strictly necessary
        # It exists to create nicer SQL code output
        inst = objects.TableInstance.make(sql.TableName(obj, obj.name), obj, [], {})
    else:
        inst = compile_remote(state, obj)

        if isinstance(inst, objects.TableInstance):
            assert isinstance(inst, objects.TableInstance)

            # Make new type
            all_aliases = []
            new_columns = {}
            for name, col in inst.columns.items():
                # ci = objects.make_column_instance(sql.Name(col.type, name), col.type, [col])
                code = sql.Name(col.type, name)
                ci = col.remake(code=code)
                new_columns[name] = ci
                all_aliases.append((col, ci))

            # Make code
            sql_fields = [
                sql.ColumnAlias.make(o.code, n.code)
                for old, new in all_aliases
                for o, n in safezip(old.flatten(), new.flatten())
            ]

            code = sql.Select(inst.type, inst.code, sql_fields)

            # Make Instance
            inst = objects.TableInstance.make(code, inst.type, [inst], new_columns)

    instances.append(inst)

    qb = sql.QueryBuilder(state.db.target, False)
    return '%s' % inst.code.compile(qb).text

import re
def pql_SQL(state: State, type_expr: ast.Expr, code_expr: ast.Expr):
    type_ = simplify(state, type_expr).concrete_type()
    sql_code = localize(state, evaluate(state, code_expr))
    assert isinstance(sql_code, str)

    # TODO escaping for security?
    instances = []
    expanded = re.sub(r"\$\w+", lambda m: _pql_SQL_callback(state, m, instances), sql_code)

    code = sql.RawSql(type_, expanded)

    # TODO validation!!
    if isinstance(type_, types.TableType):
        name = get_alias(state, "subq_")

        inst = instanciate_table(state, type_, sql.TableName(type_, name), instances)
        fields = [_make_name(path) for path, _ in inst.type.flatten()]

        subq = sql.Subquery(type_, name, fields, code)
        inst.subqueries[name] = subq

        return inst

    return objects.make_instance(code, type_, instances)

def pql_isa(state: State, expr: ast.Expr, type_expr: ast.Expr):
    inst = compile_remote(state, expr)
    type_ = simplify(state, type_expr)
    res = isinstance(inst.type.concrete_type(), type_)
    return objects.make_value_instance(res, types.Bool)

def _apply_sql_func(state, obj: ast.Expr, table_func, name):
    obj = compile_remote(state, obj)
    if isinstance(obj, objects.TableInstance):
        code = table_func(types.Int, obj.code)
    else:
        if not isinstance(obj.type.concrete_type(), types.Aggregated):
            raise pql_TypeError(None, f"Function '{name}' expected an aggregated list, but got '{obj.type.concrete_type()}' instead. Did you forget to group?")

        code = sql.FieldFunc(types.Int, name, obj.code)

    return objects.Instance.make(code, types.Int, [obj])

def pql_count(state: State, obj: ast.Expr):
    return _apply_sql_func(state, obj, sql.CountTable, 'count')

# def pql_sum(state: State, obj: ast.Expr):
#     return _apply_sql_func(state, obj, None, 'sum')


def pql_enum(state: State, table: ast.Expr):
    index_name = "index"

    table = compile_remote(state, table)

    cols = SafeDict()
    cols[index_name] = types.make_column(types.Int)
    cols.update(table.type.columns)
    new_table_type = types.TableType(get_alias(state, "enum"), cols, True)

    # Added to sqlite3 in 3.25.0: https://www.sqlite.org/windowfunctions.html
    index_code = sql.RawSql(types.Int, "row_number() over ()")
    values = [index_code] + [c.code for c in table.flatten()]

    return instanciate_table(state, new_table_type, table.code, [table], values=values)

def pql_temptable(state: State, expr: ast.Expr):
    expr = compile_remote(state, expr)
    assert isinstance(expr, objects.TableInstance)
    name = get_alias(state, "temp_" + expr.type.name)
    table = types.TableType(name, expr.type.columns, temporary=True)
    state.db.query(compile_type_def(state, table))
    state.db.query(sql.Insert(types.null, name, expr.code))
    return objects.InstancePlaceholder(table)

def pql_get_db_type(state: State):
    """
    Returns a string representing the db type that's currently connected.

    Possible values are:
        - "sqlite"
        - "postgres"
    """
    s = state.db.target
    return objects.make_instance(sql_repr(s), types.String, [])



def sql_bin_op(state, op, table1, table2, name):
    t1 = compile_remote(state, table1)
    t2 = compile_remote(state, table2)
    # TODO make sure both table types are compatiable
    l1 = len(t1.type.flatten([]))
    l2 = len(t2.type.flatten([]))
    if l1 != l2:
        raise pql_TypeError(f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    code = sql.TableArith(t1.type, op, [t1.code, t2.code])
    # TODO new type, so it won't look like the physical table
    return objects.TableInstance.make(code, t1.type, [t1, t2], t1.columns)

def pql_intersect(state, t1, t2):
    return sql_bin_op(state, "INTERSECT", t1, t2, "intersect")

def pql_substract(state, t1, t2):
    return sql_bin_op(state, "EXCEPT", t1, t2, "substract")

def pql_union(state, t1, t2):
    return sql_bin_op(state, "UNION", t1, t2, "union")

def pql_concat(state, t1, t2):
    return sql_bin_op(state, "UNION ALL", t1, t2, "concatenate")


def _join(state: State, join: str, exprs: dict, joinall=False):
    assert len(exprs) == 2
    exprs = {name: compile_remote(state, value) for name,value in exprs.items()}
    assert all(isinstance(x, objects.Instance) for x in exprs.values())

    (a,b) = exprs.values()

    if joinall:
        tables = (a,b)
    else:
        if isinstance(a, objects.ColumnInstanceWithTable) and isinstance(b, objects.ColumnInstanceWithTable):
            cols = a, b
        else:
            assert isinstance(a, objects.TableInstance) and isinstance(b, objects.TableInstance)    # TODO better error message (TypeError?)
            cols = _auto_join(state, join, a, b)
        tables = [c.table for c in cols]

    col_types = {name: types.make_column(types.StructType(name, {n:c.type.type for n, c in table.columns.items()}))
                for name, table in safezip(exprs, tables)}
    table_type = types.TableType(get_alias(state, "joinall" if joinall else "join"), SafeDict(col_types), False)

    conds = [] if joinall else [sql.Compare(types.Bool, '=', [cols[0].code, cols[1].code])]
    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    columns = dict(safezip(exprs, [t.to_struct_column() for t in tables]))
    return objects.TableInstance.make(code, table_type, [a,b], columns)

def pql_join(state, tables):
    return _join(state, "JOIN", tables)
def pql_leftjoin(state, tables):
    return _join(state, "LEFT JOIN", tables)
def pql_joinall(state: State, tables):
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
    for c in t1.columns.values():
        if isinstance(c.type, types.RelationalColumnType):
            rel = c.type.type
            if rel == t2.type:
                # TODO depends on the query
                yield (objects.ColumnInstanceWithTable(t2.get_attr('id'), t2), objects.ColumnInstanceWithTable(c, t1))

def pql_type(state: State, obj: ast.Expr):
    """
    Returns the type of the given object
    """
    inst = compile_remote(state, obj)
    t = inst.type.concrete_type()
    # if isinstance(t, types.TableType):
    #     return types.TableType
    return t

def pql_cast_int(state: State, expr: ast.Expr):
    "Temporary function, for internal use"
    # TODO make this function less ugly and better for the user
    inst = compile_remote(state, expr)
    type_ = inst.type.concrete_type()
    if isinstance(type_, types.TableType):
        assert len(inst.columns) == 1
        res = localize(state, inst)
        # res = types.Int.import_result(res)
        assert len(res) == 1
        res = list(res[0].values())[0]
        return objects.make_value_instance(res, types.Int)

    elif type_ == types.Int:
        return inst
    elif type_ == types.Float:
        value = localize(state, inst)
        return objects.make_value_instance(int(value), types.Int)

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
        return objects.make_value_instance(text, types.String)


    lines = []
    inst = compile_remote(state, obj)
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
        lines = [f"func {inst.name}({param_str})"]
        if isinstance(inst, objects.InternalFunction) and inst.func.__doc__:
            lines += [inst.func.__doc__]
    else:
        raise pql_TypeError(obj.meta, "help() only accepts functions at the moment")

    text = '\n'.join(lines)
    return objects.make_value_instance(text, types.String)

def pql_ls(state: State, obj: types.PqlObject = objects.null):
    """
    List all names in the namespace of the given object.

    If no object is given, lists the names in the current namespace.
    """
    if obj is not objects.null:
        inst = compile_remote(state, obj)
        if not isinstance(inst, objects.TableInstance):
            raise pql_TypeError(obj.meta, "Argument to ls() must be a table")
        all_vars = list(inst.columns)
    else:
        all_vars = list(state.get_all_vars())

    assert all(isinstance(s, str) for s in all_vars)
    names = [objects.make_value_instance(str(s), types.String) for s in all_vars]
    return compile_remote(state, objects.List_(None, names))



internal_funcs = {
    'help': pql_help,
    'ls': pql_ls,
    'connect': pql_connect,
    'count': pql_count,
    'enum': pql_enum,
    'temptable': pql_temptable,
    'concat': pql_concat,
    'intersect': pql_intersect,
    'union': pql_union,
    'substract': pql_substract,
    'SQL': pql_SQL,
    'isa': pql_isa,
    'type': pql_type,
    'get_db_type': pql_get_db_type,
    '_cast_int': pql_cast_int,
}
joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param(None, 'tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param(None, 'tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param(None, 'tables')),
}
