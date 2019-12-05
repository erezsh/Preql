from contextlib import contextmanager

from .dispatchy import Dispatchy
from .exceptions import pql_NameNotFound

from . import pql_ast as ast
from . import pql_objects as objects

dy = Dispatchy()

# Define common dispatch functions
@dy
def simplify():
    raise NotImplementedError()



class State:
    def __init__(self, db, fmt, ns=None):
        self.db = db
        self.fmt = fmt

        # self.ns = [_initial_namespace()]
        self.ns = ns or [{}]
        self.tick = 0

    def get_var(self, name):
        for scope in reversed(self.ns):
            if name in scope:
                return scope[name]

        try:
            meta = dict(
                line = name.line,
                column = name.column,
            )
        except AttributeError:
            meta = dict(
                line = '?',
                column = '?',
            )

        raise pql_NameNotFound(str(name), meta)

    def set_var(self, name, value):
        assert not isinstance(value, ast.Name)
        self.ns[-1][name] = value

    def get_all_vars(self):
        d = {}
        for scope in self.ns:
            d.update(scope) # Overwrite upper scopes
        return d

    def push_scope(self):
        self.ns.append({})

    def pop_scope(self):
        return self.ns.pop()


    def __copy__(self):
        s = State(self.db, self.fmt)
        s.ns = [dict(n) for n in self.ns]
        s.tick = self.tick
        return s

    @contextmanager
    def use_scope(self, scope: dict):
        x = len(self.ns)
        self.ns.append(scope)
        try:
            yield
        finally:
            self.ns.pop()
            assert x == len(self.ns)


def get_alias(state: State, obj):
    if isinstance(obj, objects.TableInstance):
        return get_alias(state, obj.type.name)

    state.tick += 1
    return obj + str(state.tick)