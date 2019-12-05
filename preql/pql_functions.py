from .utils import safezip, listgen, SafeDict
from .exceptions import pql_TypeError

from . import pql_objects as objects
from . import pql_types as types
from . import pql_ast as ast
from . import sql

from .compiler import compile_remote, instanciate_table, compile_type_def
from .interp_common import State, get_alias


def pql_limit(state: State, table: objects.TableInstance, length: objects.Instance):
    table = compile_remote(state, table)
    length = compile_remote(state, length)
    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], limit=length.code)
    return table.remake(code=code)

def _apply_sql_func(state, obj: ast.Expr, table_func, field_func):
    obj = compile_remote(state, obj)
    if isinstance(obj, objects.TableInstance):
        code = table_func(obj.type, obj.code)
    else:
        assert isinstance(obj, objects.Aggregated)
        obj = obj.expr
        assert isinstance(obj, objects.ColumnInstance), obj
        code = field_func(types.Int, obj.code)

    return objects.Instance.make(code, types.Int, [obj])

def pql_count(state: State, obj: ast.Expr):
    return _apply_sql_func(state, obj, sql.CountTable, lambda t,c: sql.FieldFunc(t, 'count', c))

def pql_sum(state: State, obj: ast.Expr):
    return _apply_sql_func(state, obj, None, lambda t,c: sql.FieldFunc(t, 'sum', c))


def pql_enum(state: State, table: ast.Expr):
    index_name = "index"

    table = compile_remote(state, table)

    new_table_type = types.TableType(get_alias(state, "enum"), {}, True)
    new_table_type.add_column(types.make_column(index_name, types.Int))
    for c in table.type.columns.values():
        new_table_type.add_column(c)

    # Added to sqlite3 in 3.25.0: https://www.sqlite.org/windowfunctions.html
    index_code = sql.RawSql(types.Int, "row_number() over ()")
    values = [index_code] + [c.code for c in table.flatten()]

    return instanciate_table(state, new_table_type, table.code, [table], values=values)

def pql_temptable(state: State, expr: ast.Expr):
    expr = compile_remote(state, expr)
    assert isinstance(expr, objects.TableInstance)
    name = get_alias(state, "temp_" + expr.type.name)
    table = types.TableType(name, expr.type.columns, temporary=True)
    state.db.query(compile_type_def(table))
    state.db.query(sql.Insert(types.null, name, expr.code))
    return table




def sql_bin_op(state, op, table1, table2, name):
    t1 = compile_remote(state, table1)
    t2 = compile_remote(state, table2)
    # TODO make sure both table types are compatiable
    l1 = len(t1.type.flatten())
    l2 = len(t2.type.flatten())
    if l1 != l2:
        raise pql_TypeError(f"Cannot {name} tables due to column mismatch (table1 has {l1} columns, table2 has {l2} columns)")

    code = sql.TableArith(t1.type, op, [t1.code, t2.code])
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

    col_types = {name: types.make_column(name, types.StructType(name, {n:c.type.type for n, c in table.columns.items()}))
                for name, table in safezip(exprs, tables)}
    table_type = types.TableType(get_alias(state, "joinall" if joinall else "join"), col_types, False)

    conds = [] if joinall else [sql.Compare(types.Bool, '==', [cols[0].code, cols[1].code])]
    code = sql.Join(table_type, join, [t.code for t in tables], conds)

    columns = dict(safezip(exprs, tables))
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


internal_funcs = {
    'limit': pql_limit,
    'count': pql_count,
    'sum': pql_sum,
    'enum': pql_enum,
    'temptable': pql_temptable,
    'concat': pql_concat,
    'intersect': pql_intersect,
    'union': pql_union,
    'substract': pql_substract,
}
joins = {
    'join': objects.InternalFunction('join', [], pql_join, objects.Param('tables')),
    'joinall': objects.InternalFunction('joinall', [], pql_joinall, objects.Param('tables')),
    'leftjoin': objects.InternalFunction('leftjoin', [], pql_leftjoin, objects.Param('tables')),
}
