from pathlib import Path


from . import exceptions as exc
from .evaluate import State, execute, eval_func_call
from .parser import parse_stmts
from . import pql_ast as ast
from . import pql_objects as objects
from .interp_common import new_value_instance

from .pql_functions import internal_funcs, joins
from .pql_types import T, from_python, Object


def initial_namespace():
    # TODO localinstance / metainstance
    ns = {k:v for k, v in T.items()}
    ns.update(internal_funcs)
    ns.update(joins)
    # TODO all exceptions
    ns['TypeError'] = exc.pql_TypeError
    ns['ValueError'] = exc.pql_ValueError
    return [dict(ns)]

class Interpreter:
    def __init__(self, sqlengine, fmt='text'):
        self.sqlengine = sqlengine
        self.state = State(self, sqlengine, fmt, initial_namespace())
        self.include('core.pql', __file__) # TODO use an import mechanism instead

    def call_func(self, fname, args):
        return eval_func_call(self.state, self.state.get_var(fname), args)

    # def eval_expr(self, code, args):
    #     expr_ast = parse_expr(code)
    #     with self.state.use_scope(args):
    #         obj = evaluate(self.state, expr_ast)
    #     return obj

    def execute_code(self, code, source_file, args=None):
        assert not args, "Not implemented yet: %s" % args
        last = None
        for stmt in parse_stmts(code, source_file):
            last = execute(self.state, stmt)
        return last

    def include(self, fn, rel_to=None):
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self.execute_code(f.read(), fn)

    def set_var(self, name, value):
        if not isinstance(value, Object):
            try:
                value = value._to_pql()
            except AttributeError:
                value = new_value_instance(value)

        self.state.set_var(name, value)

