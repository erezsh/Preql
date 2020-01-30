from contextlib import contextmanager

import dsnparse

from runtype import Dispatch

from .exceptions import pql_NameNotFound, pql_TypeError, Meta

from . import pql_ast as ast
from . import pql_objects as objects
from . import pql_types as types
from . import sql
from .sql_interface import SqliteInterface, PostgresInterface

dy = Dispatch()

# Define common dispatch functions
@dy
def simplify():
    raise NotImplementedError()

class GlobalSettings:
    Optimize = True


class State:
    def __init__(self, db, fmt, ns=None):
        self.db = db
        self.fmt = fmt

        self.ns = ns or [{}]
        self.tick = 0

    def get_var(self, name):
        for scope in reversed(self.ns):
            if name in scope:
                return scope[name]

        raise pql_NameNotFound(getattr(name, 'meta', None), str(name))

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

    def connect(self, uri):
        print(f"[Preql] Connecting to {uri}")
        self.db = create_engine(uri, self.db._debug)


def create_engine(db_uri, debug):
    dsn = dsnparse.parse(db_uri)
    if len(dsn.paths) != 1:
        raise ValueError("Bad value for uri: %s" % db_uri)
    path ,= dsn.paths
    if dsn.scheme == 'sqlite':
        return SqliteInterface(path, debug=debug)
    elif dsn.scheme == 'postgres':
        return PostgresInterface(dsn.host, path, dsn.user, dsn.password, debug=debug)

    raise NotImplementedError(f"Scheme {dsn.scheme} currently not supported")



def get_alias(state: State, obj):
    if isinstance(obj, objects.TableInstance):
        return get_alias(state, obj.type.name)

    state.tick += 1
    return obj + str(state.tick)


def assert_type(meta, t, type_, msg):
    if not isinstance(t, type_):
        raise pql_TypeError(meta, msg % (type_, t))


def sql_repr(x):
    if x is None:
        return sql.null

    t = types.Primitive.by_pytype[type(x)]
    if t is types.DateTime:
        # TODO Better to pass the object instead of a string?
        return sql.Primitive(t, repr(str(x)))

    if t is types.String or t is types.Text:
        return sql.Primitive(t, "'%s'" % str(x).replace("'", "''"))

    return sql.Primitive(t, repr(x))

def make_value_instance(value, type_=None, force_type=False):
    r = sql_repr(value)
    if force_type:
        assert type_
    elif type_:
        assert isinstance(type_, (types.Primitive, types.NullType, types.IdType)), type_
        assert r.type == type_, (r.type, type_)
    else:
        type_ = r.type
    if GlobalSettings.Optimize:
        return objects.ValueInstance.make(r, type_, [], value)
    else:
        return objects.Instance.make(r, type_, [])


def python_to_pql(value):
    # TODO why not just make value instance?
    if value is None:
        return types.null
    elif isinstance(value, str):
        return ast.Const(None, types.String, value)
    elif isinstance(value, int):
        return ast.Const(None, types.Int, value)
    elif isinstance(value, list):
        # return ast.Const(None, types.ListType(types.String), value)
        return objects.List_(None, list(map(python_to_pql, value)))
    assert False, value

