from typing import Optional

from .utils import safezip, listgen, SafeDict
from .exceptions import pql_TypeError, pql_JoinError

from . import pql_objects as objects
from . import pql_types as types
from . import pql_ast as ast
from . import sql

from .compiler import compile_remote, instanciate_table, compile_type_def, alias_table, exclude_fields, rename_field
from .interp_common import State, get_alias, make_value_instance, dy
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

            # Make new type
            new_columns = {
                name: objects.make_column_instance(sql.Name(col.type, name), col.type, [col])
                for name, col in inst.columns.items()
            }

            # Make code
            sql_fields = [
                sql.ColumnAlias.make(o.code, n.code)
                for old, new in safezip(inst.columns.values(), new_columns.values())
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
    type_ = simplify(state, type_expr)
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
        # TODO this isn't in the tests!
        fields = [sql.Name(c, path) for path, c in inst.type.flatten_type()]

        subq = sql.Subquery(type_, name, fields, code)
        inst.subqueries[name] = subq

        return inst

    return objects.Instance.make(code, type_, instances)

def pql_breakpoint(state: State):
    # breakpoint()
    state._py_api.start_repl()

def pql_isa(state: State, expr: ast.Expr, type_expr: ast.Expr):
    inst = compile_remote(state, expr)
    type_ = simplify(state, type_expr)
    res = isinstance(inst.type, type_)
    return make_value_instance(res, types.Bool)

def _count(state, obj: ast.Expr, table_func, name):
    obj = compile_remote(state, obj)

    if isinstance(obj, objects.TableInstance):
        code = table_func(types.Int, obj.code)
    else:
        if not isinstance(obj.type, types.Aggregated):
            raise pql_TypeError(None, f"Function '{name}' expected an aggregated list, but got '{obj.type}' instead. Did you forget to group?")

        if isinstance(obj, objects.StructColumnInstance):
            # XXX Counting a struct means counting its id
            # But what happens if there is no 'id'?
            obj = obj.get_attr('id')

        code = sql.FieldFunc(types.Int, name, obj.code)

    return objects.Instance.make(code, types.Int, [obj])

def pql_count(state: State, obj: ast.Expr):
    return _count(state, obj, sql.CountTable, 'count')


def pql_temptable(state: State, expr: ast.Expr):
    expr = compile_remote(state, expr)
    assert isinstance(expr, objects.TableInstance)

    name = get_alias(state, "temp_" + expr.type.name)
    table = types.TableType(name, expr.type.columns, temporary=True, primary_keys=[['id']])
    if 'id' not in table.columns:
        table.columns['id'] = types.IdType(table)

    state.db.query(compile_type_def(state, table))

    # if table.flat_length() != expr.type.flat_length():
    #     assert False
    primary_keys, columns = table.flat_for_insert()
    expr = exclude_fields(state, expr, primary_keys)
    # expr = rename_field(state, expr, primary_keys)
    state.db.query(sql.Insert(types.null, table, columns, expr.code), expr.subqueries)

    return instanciate_table(state, table, sql.TableName(table, table.name), [])

def pql_get_db_type(state: State):
    """
    Returns a string representing the db type that's currently connected.

    Possible values are:
        - "sqlite"
        - "postgres"
    """
    return make_value_instance(state.db.target, types.String)



def sql_bin_op(state, op, table1, table2, name):
    t1 = compile_remote(state, table1)
    t2 = compile_remote(state, table2)
    # TODO make sure both table types are compatiable
    l1 = len(t1.type.flatten_type())
    l2 = len(t2.type.flatten_type())
    if l1 != l2:
        raise pql_TypeError(None, f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

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



def _join(state: State, join: str, exprs: dict, joinall=False, nullable=None):
    assert len(exprs) == 2
    exprs = {name: compile_remote(state, value) for name,value in exprs.items()}
    assert all(isinstance(x, objects.Instance) for x in exprs.values())

    (a,b) = exprs.values()

    if a is objects.EmptyList or b is objects.EmptyList:
        raise pql_TypeError(None, "Cannot join on an untyped empty list")

    if isinstance(a, objects.ColumnReference) and isinstance(b, objects.ColumnReference):
        a = a.replace(table=alias_table(state, a.table))
        b = b.replace(table=alias_table(state, b.table))
        cols = a, b
        tables = [a.table, b.table]
    else:
        if not (isinstance(a, objects.TableInstance) and isinstance(b, objects.TableInstance)):
            raise pql_TypeError(None, f"join() got unexpected values:\n * {a}\n * {b}")
        a = alias_table(state, a)
        b = alias_table(state, b)
        if joinall:
            tables = (a, b)
        else:
            cols = _auto_join(state, join, a, b)
            tables = [c.table for c in cols]

    col_types = {name: types.StructType(name, {n:c.type for n, c in table.columns.items()})
                for name, table in safezip(exprs, tables)}

    # Update nullable for left/right/outer joins
    if nullable:
        col_types = {name: types.OptionalType(t) if n else t
                    for (name, t), n in safezip(col_types.items(), nullable)}

    primary_keys = [ [name] + pk
                        for name, t in safezip(exprs, tables)
                        for pk in t.type.primary_keys
                    ]
    table_type = types.TableType(get_alias(state, "joinall" if joinall else "join"), SafeDict(col_types), False, primary_keys)

    conds = [] if joinall else [sql.Compare(types.Bool, '=', [cols[0].code, cols[1].code])]

    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    columns = dict(safezip(exprs, [t.to_struct_column() for t in tables]))
    return objects.TableInstance.make(code, table_type, [a,b], columns)

def pql_join(state, tables):
    return _join(state, "JOIN", tables)
def pql_leftjoin(state, tables):
    return _join(state, "LEFT JOIN", tables, nullable=[False, True])
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
    for name, c in t1.columns.items():
        if isinstance(c.type, types.RelationalColumn):
            rel = c.type.type
            if rel == t2.type:
                # TODO depends on the query
                yield (objects.ColumnReference(t2, 'id'), objects.ColumnReference(t1, name))

def pql_type(state: State, obj: ast.Expr):
    """
    Returns the type of the given object
    """
    inst = compile_remote(state, obj)
    t = inst.type   # XXX concrete?
    # if isinstance(t, types.TableType):
    #     return types.TableType
    return t

def pql_cast(state: State, obj: ast.Expr, type_: ast.Expr):
    inst = compile_remote(state, obj)
    type_ = compile_remote(state, type_)
    if not isinstance(type_, types.PqlType):
        raise pql_TypeError(type_.meta, f"Cast expected a type, got {type_} instead.")

    return _cast(state, inst.type, type_, inst)

@dy
def _cast(state, inst_type: types.ListType, target_type: types.ListType, inst):
    if inst_type == target_type:
        return inst

    if inst is objects.EmptyList:
        return compile_remote(state, ast.List_(None, []), target_type.elemtype)

    raise pql_TypeError(None, "Cast not fully implemented yet")


def pql_cast_int(state: State, expr: ast.Expr):
    "Temporary function, for internal use"
    # TODO make this function less ugly and better for the user
    inst = compile_remote(state, expr)
    type_ = inst.type
    if isinstance(type_, types.TableType):
        assert len(inst.columns) == 1
        res = localize(state, inst)
        # res = types.Int.import_result(res)
        assert len(res) == 1
        res = list(res[0].values())[0]
        return make_value_instance(res, types.Int)

    elif type_ == types.Int:
        return inst
    elif isinstance(type_, types.IdType):
        return inst.replace(type=types.Int)
    elif type_ == types.Float:
        value = localize(state, inst)
        return make_value_instance(int(value), types.Int)

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
        return make_value_instance(text, types.String)


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
    return make_value_instance(text, types.String)

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
    names = [make_value_instance(str(s), types.String) for s in all_vars]
    return compile_remote(state, ast.List_(None, names))



internal_funcs = {
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
    'isa': pql_isa,
    'type': pql_type,
    'breakpoint': pql_breakpoint,
    'get_db_type': pql_get_db_type,
    '_cast_int': pql_cast_int,
    'cast': pql_cast,
}
joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param(None, 'tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param(None, 'tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param(None, 'tables')),
}
