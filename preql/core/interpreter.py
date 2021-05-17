from copy import copy
import threading
from pathlib import Path
from functools import wraps

from preql.utils import classify
from preql.context import context

from .exceptions import Signal, pql_SyntaxError, ReturnSignal
from .evaluate import execute, eval_func_call, import_module, evaluate, cast_to_python
from .parser import parse_stmts
from . import pql_ast as ast
from . import pql_objects as objects
from .interp_common import pyvalue_inst, call_builtin_func
from .state import ThreadState
from .pql_types import T, Object
from .pql_functions import import_pandas
from .pql_functions import internal_funcs, joins


def initial_namespace():
    # TODO localinstance / metainstance
    ns = {k:v for k, v in T.items()}
    ns.update(internal_funcs)
    ns.update(joins)
    # TODO all exceptions
    name = '__builtins__'
    module = objects.Module(name, dict(ns))
    return [{name: module}]


def entrypoint(f):
    @wraps(f)
    def inner(interp, *args, **kwargs):
        with interp.setup_context():
            return f(interp, *args, **kwargs)
    return inner


class LocalCopy(threading.local):
    def __init__(self, **kw):
        self._items = kw

    def __getattr__(self, attr):
        # Only runs the first time, due to setattr
        value = copy(self._items[attr])
        setattr(self, attr, value)
        return value


class Interpreter:
    def __init__(self, sqlengine, display, use_core=True):
        self.state = ThreadState.from_components(self, sqlengine, display, initial_namespace())
        if use_core:
            mns = import_module(self.state, ast.Import('__builtins__', use_core=False)).namespace
            bns = self.state.get_var('__builtins__').namespace
            # safe-update
            for k, v in mns.items():
                if not k.startswith('__'):
                    assert k not in bns
                    bns[k] = v

        self._local_copies = LocalCopy(state=self.state)

    def setup_context(self):
        return context(state=self._local_copies.state)

    def _execute_code(self, code, source_file, args=None):
        # assert not args, "Not implemented yet: %s" % args
        try:
            stmts = parse_stmts(code, source_file)
        except pql_SyntaxError as e:
            raise Signal(T.SyntaxError, [e.text_ref], e.message)


        if stmts:
            if isinstance(stmts[0], ast.Const) and stmts[0].type == T.string:
                self.set_var('__doc__', stmts[0].value)

        last = None

        # with self.state.ns.use_parameters(args or {}):
        with context(parameters=args or {}):    # Set parameters for Namespace.get_var()
            for stmt in stmts:
                try:
                    last = execute(stmt)
                except ReturnSignal:
                    raise Signal.make(T.CodeError, stmt, "'return' outside of function")

        return last

    def _include(self, fn, rel_to=None):
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self._execute_code(f.read(), fn)

    def set_var(self, name, value):
        if not isinstance(value, Object):
            try:
                value = value._to_pql()
            except AttributeError:
                value = pyvalue_inst(value)

        self.state.set_var(name, value)

    def has_var(self, name):
        try:
            self.state.get_var(name)
        except Signal:
            return False
        return True



    #####################

    execute_code = entrypoint(_execute_code)
    include = entrypoint(_include)


    @entrypoint
    def evaluate_obj(self, obj):
        return evaluate(obj)

    @entrypoint
    def localize_obj(self, obj):
        return obj.localize()

    @entrypoint
    def call_func(self, fname, args, kw=None):
        if kw:
            args = args + [ast.NamedField(k, v, False) for k,v in kw.items()]
        res = eval_func_call(context.state.get_var(fname), args)
        return evaluate(res)

    @entrypoint
    def cast_to_python(self, obj):
        return cast_to_python(obj)

    @entrypoint
    def call_builtin_func(self, name, args):
        return call_builtin_func(name, args)

    @entrypoint
    def import_pandas(self, dfs):
        return list(import_pandas(dfs))

    @entrypoint
    def list_tables(self):
        return self.state.db.list_tables()


    @entrypoint
    def load_all_tables(self):
        table_types = self.state.db.import_table_types()
        table_types_by_schema = classify(table_types, lambda x: x[0], lambda x: x[1:])

        for schema_name, table_types in table_types_by_schema.items():
            if schema_name:
                schema = objects.Module(schema_name, {})
                self.set_var(schema_name, schema)

            for table_name, table_type in table_types:
                db_name = table_type.options['name']
                inst = objects.new_table(table_type, db_name)

                if schema_name:
                    schema.namespace[table_name] = inst
                else:
                    if not self.has_var(table_name):
                        self.set_var(table_name, inst)


    def clone(self, use_core):
        state = self.state
        i = Interpreter(state.db, state.display, use_core=use_core)
        i.state.stacktrace = state.stacktrace   # XXX proper interface
        return i