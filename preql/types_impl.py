from datetime import datetime
import json
from .exceptions import Signal, pql_AttributeError

from .base import Object
from .utils import listgen, concat_for, classify_bool
from .pql_types import ITEM_NAME, T, Type, dp_type, dp_inst, from_python


def Object_repr(self, state):
    return repr(self)

def Object_get_attr(self, attr):
    raise pql_AttributeError(attr)

def Object_isa(self, t):
    if not isinstance(t, Type):
        raise Signal.make(T.TypeError, [], f"'type' argument to isa() isn't a type. It is {t}")
    return self.type <= t

Object.repr = Object_repr
Object.get_attr = Object_get_attr
Object.isa = Object_isa


@dp_type
def flatten_path(path, t):
    return [(path, t)]

@dp_type
def flatten_path(path, t: T.union[T.table, T.struct]):
    elems = t.elems
    if t.maybe_null():
        elems = {k:v.as_nullable() for k, v in elems.items()}
    return concat_for(flatten_path(path + [name], col) for name, col in elems.items())
@dp_type
def flatten_path(path, t: T.list):
    return concat_for(flatten_path(path + [name], col) for name, col in [(ITEM_NAME, t.elem)])


def flatten_type(tp, path = []):
    # return [(join_names(path), t.col_type) for path, t in flatten_path([name], tp)]
    return [(join_names(path), t) for path, t in flatten_path(path, tp)]


def table_params(t):
    # TODO hide_from_init / writeable, not id
    return [(name, c) for name, c in t.elems.items() if not c <= T.t_id]



def table_flat_for_insert(table):
    # auto_count = join_names(self.primary_keys)
    # if 'pk' not in table.options:
    #     raise Signal.pql_TypeError(T.TypeError, [], f"Cannot add to table. Primary key not defined")

    pks = {join_names(pk) for pk in table.options.get('pk', [])}
    names = [name for name, t in flatten_type(table)]
    return classify_bool(names, lambda name: name in pks)


def repr_value(state, v):
    return pql_repr(state, v.type, v.value)

@dp_type
def pql_repr(state, t: T.object, value):
    return repr(value)

@dp_type
def pql_repr(state, t: T.function, value):
    params = []
    for p in value.params:
        s = p.name
        if p.type:
            s += ": %s" % p.type
        params.append(s)

    return f'{{func {value.name}({", ".join(params)})}}'

@dp_type
def pql_repr(state, t: T.decimal, value):
    raise Signal.make(T.NotImplementedError, state, None, "Decimal not implemented")

@dp_type
def pql_repr(state, t: T.string, value):
    assert isinstance(value, str), value
    value = value.replace('"', r'\"')
    return f'"{value}"'

@dp_type
def pql_repr(state, t: T.text, value):
    assert isinstance(value, str), value
    return str(value)

@dp_type
def pql_repr(state, t: T.bool, value):
    return 'true' if value else 'false'

@dp_type
def pql_repr(state, t: T.nulltype, value):
    return 'null'


@dp_inst
def from_sql(state, res: T.primitive):
    try:
        row ,= res.value
        item ,= row
    except ValueError:
        raise Signal.make(T.TypeError, state, None, "Expected primitive. Got: '%s'" % res.value)
    # t = from_python(type(item))
    # if not (t <= res.type):
    #     raise Signal.make(T.TypeError, state, None, f"Incorrect type returned from SQL: '{t}' instead of '{res.type}'")
    return item

def _from_datetime(state, s):
    if s is None:
        return None

    # Postgres
    if isinstance(s, datetime):
        return s

    # Sqlite
    if not isinstance(s, str):
        raise Signal.make(T.TypeError, [], f"datetime expected a string. Instead got: {s}")
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise Signal.make(T.ValueError, state, None, str(e))

@dp_inst
def from_sql(state, res: T.datetime):
    # XXX doesn't belong here?
    row ,= res.value
    item ,= row
    s = item
    return _from_datetime(state, s)

@dp_inst
def from_sql(state, arr: T.list):
    if not all(len(e)==1 for e in arr.value):
        raise Signal.make(T.TypeError, state, None, f"Expected 1 column. Got {len(arr.value[0])}")
    return [e[0] for e in arr.value]

@dp_inst
@listgen
def from_sql(state, arr: T.table):
    expected_length = len(flatten_type(arr.type))   # TODO optimize?
    for row in arr.value:
        if len(row) != expected_length:
            raise Signal.make(T.TypeError, state, None, f"Expected {expected_length} columns, but got {len(row)}")
        i = iter(row)
        yield {name: restructure_result(state, col, i) for name, col in arr.type.elems.items()}

@dp_type
def restructure_result(state, t: T.table, i):
    # return ({name: restructure_result(state, col, i) for name, col in t.elem_dict.items()})
    return next(i)

@dp_type
def restructure_result(state, t: T.struct, i):
    return ({name: restructure_result(state, col, i) for name, col in t.elems.items()})

@dp_type
def restructure_result(state, t: T.union[T.primitive, T.nulltype], i):
    return next(i)

@dp_type
def restructure_result(state, t: T.vectorized[T.union[T.primitive, T.nulltype]], i):
    return next(i)


@dp_type
def restructure_result(state, t: T.list[T.union[T.primitive, T.nulltype]], i):
    # XXX specific to choice of db. So belongs in sql.py?
    res = next(i)
    if state.db.target == 'mysql':   # TODO use constant
        res = json.loads(res)
    elif state.db.target == 'sqlite':
        res = res.split('|')
    if t.elem <= T.int: # XXX hack! TODO Use a generic form
        res = [int(x) for x in res]
    elif t.elem <= T.float: # XXX hack! TODO Use a generic form
        res = [float(x) for x in res]
    return res

@dp_type
def restructure_result(state, t: T.datetime, i):
    s = next(i)
    return _from_datetime(None, s)


def join_names(names):
    return "_".join(names)

