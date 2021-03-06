from contextlib import contextmanager
from copy import copy
from logging import getLogger

from preql.utils import dsp
from preql.sql_interface import ConnectError, create_engine

from . import pql_ast as ast
from . import pql_objects as objects
from .exceptions import Signal
from .exceptions import InsufficientAccessLevel
from .pql_types import Type, T

logger = getLogger('interp')

# Define common dispatch functions
@dsp
def simplify(state, obj: type(NotImplemented)) -> object:
    raise NotImplementedError()

@dsp
def evaluate(state, obj: type(NotImplemented)) -> object:
    raise NotImplementedError()

@dsp
def cast_to_python(state, obj: type(NotImplemented)) -> object:
    raise NotImplementedError(obj)



class AccessLevels:
    COMPILE = 1
    # COMPILE_TEXT
    QUERY = 2
    EVALUATE = 3
    READ_DB = 4
    WRITE_DB = 5


class State:
    AccessLevels = AccessLevels

    def __init__(self, interp, db, display, ns=None):
        self.db = db
        self.interp = interp
        self.display = display
        # Add logger?

        self.ns = Namespace(ns)
        self.tick = [0]

        self.access_level = AccessLevels.WRITE_DB
        self._cache = {}
        self.stacktrace = []

    @classmethod
    def clone(cls, inst):
        s = cls(inst.interp, inst.db, inst.display)
        s.ns = copy(inst.ns)
        s.tick = inst.tick
        s.access_level = inst.access_level
        s._cache = inst._cache
        s.stacktrace = copy(inst.stacktrace)
        return s

    def __copy__(self):
        return self.clone(self)

    def limit_access(self, new_level):
        return self.reduce_access(min(new_level, self.access_level))

    def reduce_access(self, new_level):
        assert new_level <= self.access_level
        s = copy(self)
        s.access_level = new_level
        return s

    def require_access(self, level):
        if self.access_level < level:
            raise InsufficientAccessLevel(level)
    def catch_access(self, level):
        if self.access_level < level:
            raise Exception("Bad access. Security risk.")

    def connect(self, uri, auto_create=False):
        logger.info(f"[Preql] Connecting to {uri}")
        try:
            self.db = create_engine(uri, self.db._print_sql, auto_create)
        except NotImplementedError as e:
            raise Signal.make(T.NotImplementedError, None, *e.args) from e
        except ConnectError as e:
            raise Signal.make(T.DbConnectionError, None, *e.args) from e
        except ValueError as e:
            raise Signal.make(T.ValueError, None, *e.args) from e

        self._db_uri = uri


    def get_all_vars(self):
        return self.ns.get_all_vars()

    def get_all_vars_with_rank(self):
        return self.ns.get_all_vars_with_rank()

    def get_var(self, name):
        try:
            return self.ns.get_var(name)
        except NameNotFound:
            builtins = self.ns.get_var('__builtins__')
            assert isinstance(builtins, objects.Module)
            try:
                return builtins.namespace[name]
            except KeyError:
                pass

            raise Signal.make(T.NameError, name, f"Name '{name}' is not defined")


    def set_var(self, name, value):
        return self.ns.set_var(name, value)

    def use_scope(self, scope: dict):
        return self.ns.use_scope(scope)

    def unique_name(self, obj):
        self.tick[0] += 1
        return obj + str(self.tick[0])


class NameNotFound(Exception):
    pass

class Namespace:
    def __init__(self, ns=None):
        self._ns = ns or [{}]

    def __copy__(self):
        return Namespace([dict(n) for n in self._ns])

    def get_var(self, name):
        for scope in reversed(self._ns):
            if name in scope:
                return scope[name]

        raise NameNotFound(name)

    def set_var(self, name, value):
        assert not isinstance(value, ast.Name)
        self._ns[-1][name] = value


    @contextmanager
    def use_scope(self, scope: dict):
        x = len(self._ns)
        self._ns.append(scope)
        try:
            yield
        finally:
            _discarded_scope = self._ns.pop()
            assert x == len(self._ns)


    # def push_scope(self):
    #     self.ns.append({})

    # def pop_scope(self):
    #     return self.ns.pop()

    def __len__(self):
        return len(self._ns)

    def get_all_vars(self):
        d = {}
        for scope in reversed(self._ns):
            d.update(scope) # Overwrite upper scopes
        return d

    def get_all_vars_with_rank(self):
        d = {}
        for i, scope in enumerate(reversed(self._ns)):
            for k, v in scope.items():
                if k not in d:
                    d[k] = i, v
        return d



def assert_type(t, type_, ast_node, op, msg="%s expected an object of type %s, instead got '%s'"):
    assert isinstance(t, Type), t
    assert isinstance(type_, Type)
    if not t <= type_:
        if type_.typename == 'union':
            type_str = ' or '.join("'%s'" % elem for elem in type_.elems)
        else:
            type_str = "'%s'" % type_
        raise Signal.make(T.TypeError, ast_node, msg % (op, type_str, t))

def exclude_fields(state, table, fields):
    proj = ast.Projection(table, [ast.NamedField(None, ast.Ellipsis(None, exclude=list(fields) ), user_defined=False)])
    return evaluate(state, proj)

def call_builtin_func(state, name, args):
    "Call a builtin pql function"
    builtins = state.ns.get_var('__builtins__')
    assert isinstance(builtins, objects.Module)

    expr = ast.FuncCall(builtins.namespace[name], args)
    return evaluate(state, expr)



def is_global_scope(state):
    assert len(state.ns) != 0
    return len(state.ns) == 1


# def cast_to_python_primitive(state, obj):
#     res = cast_to_python(state, obj)
#     assert isinstance(res, (int, str, float, dict, list, type(None), datetime)), (res, type(res))
#     return res

def cast_to_python_string(state, obj: objects.AbsInstance):
    res = cast_to_python(state, obj)
    if not isinstance(res, str):
        raise Signal.make(T.TypeError, obj, f"Expected string, got '{res}'")
    return res

def cast_to_python_int(state, obj: objects.AbsInstance):
    res = cast_to_python(state, obj)
    if not isinstance(res, int):
        raise Signal.make(T.TypeError, obj, f"Expected string, got '{res}'")
    return res

    
pyvalue_inst = objects.pyvalue_inst

