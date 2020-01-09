from pathlib import Path

from .utils import SafeDict
from .exceptions import PreqlError, pql_TypeError, pql_ValueError

from .evaluate import State, execute, evaluate, simplify, localize
from .parser import parse_stmts, parse_expr
from . import pql_ast as ast
from . import pql_objects as objects
from . import pql_types as types
from .interp_common import make_value_instance

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
    ns['ValueError'] = pql_ValueError
    return [dict(ns)]

class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State(sqlengine, 'text', initial_namespace())
        self.include('core.pql', __file__) # TODO use an import mechanism instead

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

    def include(self, fn, rel_to=None):
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self.execute_code(f.read())

    def set_var(self, name, value):
        if not isinstance(value, types.PqlObject):
            try:
                value = value._to_pql()
            except AttributeError:
                value = make_value_instance(value)

        self.state.set_var(name, value)