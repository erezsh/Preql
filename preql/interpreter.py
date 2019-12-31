from .utils import SafeDict
from .exceptions import PreqlError, pql_TypeError

from .evaluate import State, execute, evaluate, simplify, localize
from .parser import parse_stmts, parse_expr
from . import pql_ast as ast
from . import pql_objects as objects
from . import pql_types as types

from .pql_functions import internal_funcs, joins

import inspect
def _canonize_default(d):
    return None if d is inspect._empty else d

def _create_internal_func(fname, f):
    sig = inspect.signature(f)
    return objects.InternalFunction(fname, [
            objects.Param(None, pname, type_, _canonize_default(sig.parameters[pname].default))
            for pname, type_ in list(f.__annotations__.items())[1:]
    ], f)

def initial_namespace():
    ns = SafeDict({p.name: p for p in types.primitives_by_pytype.values()})
    ns.update({
        fname: _create_internal_func(fname, f) for fname, f in internal_funcs.items()
    })
    ns.update(joins)
    ns['list'] = types.ListType
    ns['aggregate'] = types.Aggregated
    ns['TypeError'] = pql_TypeError
    return [dict(ns)]

class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State(sqlengine, 'text', initial_namespace())
        self.execute_code("""
            func _sql_agg_func(name, field)
                if isa(field, aggregate)
                    return SQL(int, name + "($field)")
                else
                    if isa(field, list)
                        return int( field{ => value: _sql_agg_func(name, value) } )  # Recursive
                    end
                end
                throw new TypeError(name + "() doesn't support field of type '" + (type(field).__name__) + "'")
            end
            func sum(field) = _sql_agg_func("SUM", field)
            func min(field) = _sql_agg_func("MIN", field)
            func max(field) = _sql_agg_func("MAX", field)
            func limit(table, lim) = SQL(type(table), "SELECT * FROM $table LIMIT $lim")
            func limit_offset(table, lim, offset) = SQL(type(table), "SELECT * FROM $table LIMIT $lim OFFSET $offset")

            if get_db_type() == "postgres"
                func repeat(s, num) = SQL(string, "REPEAT($s, $num)")
                func now() = SQL(datetime, "NOW()") # Postgres
            else
                if get_db_type() == "sqlite"
                    func repeat(s, num) = SQL(string, "replace(hex(zeroblob($num)), '00', $s)")
                    func now() = SQL(datetime, "datetime('now')") # Sqlite
                else
                    throw new TypeError("Unexpected")
                end
            end

            func sample_ratio(tbl, ratio) = SQL(tbl, "(SELECT * FROM flights where abs(CAST(random() AS REAL))/9223372036854775808 < $ratio)")

            func sample_fast(tbl, size)
                c = count(tbl)
                if size >= c
                    throw new ValueError("Asking for a sample size larger than the table")
                end
                if c == 0
                    return tbl
                end

                results = temptable(limit(sample_ratio(tbl, 1.05 * size / c), size))
                if count(results) == size
                    return results
                end
                return results + limit(tbl, size - count(results))
            end


        """)

    def call_func(self, fname, args):
        obj = simplify(self.state, ast.Name(None, fname))
        if isinstance(obj, objects.TableInstance):
            assert not args, args
            # return localize(self.state, obj)
            return obj

        funccall = ast.FuncCall(None, ast.Name(None, fname), args)
        return evaluate(self.state, funccall)

    def eval_expr(self, code, args):
        expr_ast = parse_expr(code)
        with self.state.use_scope(args):
            obj = evaluate(self.state, expr_ast)
        return obj

    def execute_code(self, code, args=None):
        assert not args, "Not implemented yet: %s" % args
        last = None
        for stmt in parse_stmts(code):
            last = execute(self.state, stmt)
        return last
