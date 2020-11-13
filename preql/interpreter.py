from pathlib import Path


from .exceptions import Signal, pql_SyntaxError
from .evaluate import State, execute, eval_func_call, import_module
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
    name = '__builtins__'
    module = objects.Module(name, dict(ns))
    return [{name: module}]



class Interpreter:
    def __init__(self, sqlengine, fmt='text', use_core=True):
        self.state = State(self, sqlengine, fmt, initial_namespace())
        if use_core:
            mns = import_module(self.state, ast.Import('__builtins__', use_core=False)).namespace
            bns = self.state.get_var('__builtins__').namespace
            # safe-update
            for k, v in mns.items():
                assert k not in bns
                bns[k] = v

    def call_func(self, fname, args):
        return eval_func_call(self.state, self.state.get_var(fname), args)

    def execute_code(self, code, source_file, args=None):
        assert not args, "Not implemented yet: %s" % args
        last = None
        try:
            stmts = parse_stmts(code, source_file)
        except pql_SyntaxError as e:
            raise Signal(T.SyntaxError, [e.text_ref], e.message)

        for stmt in stmts:
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

    def has_var(self, name):
        try:
            self.state.get_var(name)
        except Signal:
            return False
        return True

