from logging import getLogger
from copy import copy
from contextlib import contextmanager

from preql.context import context


from .exceptions import InsufficientAccessLevel, Signal
from .pql_types import T, Id
from . import pql_ast as ast

class NameNotFound(Exception):
    pass


logger = getLogger('state')

class Namespace:
    def __init__(self, ns=None):
        self._ns = ns or [{}]
        # self._parameters = None

    def __copy__(self):
        return Namespace([self._ns[0]] + [dict(n) for n in self._ns[1:]])     # Shared global namespace

    def get_var(self, name):
        # Uses context.parameters, set in Interpreter
        parameters = getattr(context, 'parameters', None)
        if parameters and name in parameters:
            return parameters[name]

        for scope in reversed(self._ns):
            if name in scope:
                return scope[name]

        raise NameNotFound(name)

    def set_var(self, name, value):
        if isinstance(name, Id):
            path = name.parts[:-1]
            ns = self
            for p in path:
                ns = ns.get_var(p)
            ns.set_var(name.parts[-1], value)
            return

        assert isinstance(name, str), name
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


class AccessLevels:
    COMPILE = 1
    # COMPILE_TEXT
    QUERY = 2
    EVALUATE = 3
    READ_DB = 4
    WRITE_DB = 5


class State:

    def __init__(self, interp, db, display, ns=None):
        self.db = db
        self.interp = interp
        self.display = display
        # Add logger?

        self.tick = [0]

        self._cache = {}


    def connect(self, uri, auto_create=False):
        from preql.sql_interface import ConnectError, create_engine
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


    def unique_name(self, obj):
        self.tick[0] += 1
        return obj + str(self.tick[0])


class ThreadState:
    def __init__(self, state, ns=None):
        self.state = state
        # Add logger?

        self.ns = Namespace(ns)
        self.access_level = AccessLevels.WRITE_DB
        self.stacktrace = []

    @classmethod
    def from_components(cls, interp, db, display, ns=None):
        state = State(interp, db, display)
        return cls(state, ns)

    @property
    def interp(self):
        return self.state.interp

    @property
    def db(self):
        return self.state.db

    @property
    def display(self):
        return self.state.display
    

    @classmethod
    def clone(cls, inst):
        s = cls(inst.state)
        s.ns = copy(inst.ns)
        s.access_level = inst.access_level
        s.stacktrace = copy(inst.stacktrace)
        return s


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


    def get_all_vars(self):
        return self.ns.get_all_vars()

    def get_all_vars_with_rank(self):
        return self.ns.get_all_vars_with_rank()

    def has_var(self, name):
        try:
            self.ns.get_var(name)
        except NameNotFound:
            return False
            
        return True

    def get_var(self, name):
        try:
            return self.ns.get_var(name)
        except NameNotFound:
            builtins = self.ns.get_var('__builtins__')
            assert builtins.type <= T.module
            try:
                return builtins.namespace[name]
            except KeyError:
                pass

            raise Signal.make(T.NameError, name, f"Name '{name!r}' is not defined")     # TODO XXX: str(name) fails under mysql


    def set_var(self, name, value):
        try:
            return self.ns.set_var(name, value)
        except NameNotFound as e:
            raise Signal.make(T.NameError, None, str(e))

    def use_scope(self, scope: dict):
        return self.ns.use_scope(scope)

    def __copy__(self):
        return self.clone(self)

    def connect(self, uri, auto_create=False):
        return self.state.connect(uri, auto_create)

    def unique_name(self, obj):
        return self.state.unique_name(obj)



def set_var(name, value):
    return context.state.set_var(name, value)

def use_scope(scope):
    return context.state.use_scope(scope)

def get_var(name):
    return context.state.get_var(name)

def has_var(name):
    return context.state.has_var(name)

def get_db():
    return context.state.db

def unique_name(prefix):
    return context.state.unique_name(prefix)

def require_access(access):
	return context.state.require_access(access)

def catch_access(access):
	return context.state.catch_access(access)

def get_access_level():
	return context.state.access_level

def reduce_access(new_level):
	return context.state.reduce_access(new_level)

def get_display():
	return context.state.display
