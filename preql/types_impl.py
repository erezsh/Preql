import json

from .base import Object
from . import exceptions as exc
from .utils import dataclass, listgen, concat_for, classify_bool
from .pql_types import T, Type, dp_type, dp_inst


def Object_repr(self, state):
    return repr(self)

def Object_get_attr(self, attr):
    raise exc.pql_AttributeError([], f"{self} has no attribute: {attr}")    # XXX TODO

def Object_isa(self, t):
    if not isinstance(t, Type):
        raise exc.pql_TypeError([], f"'type' argument to isa() isn't a type. It is {t}")
    return self.type <= t

Object.repr = Object_repr
Object.get_attr = Object_get_attr
Object.isa = Object_isa


@dp_type
def flatten_path(path, t):
    return [(path, t)]

@dp_type
def flatten_path(path, t: T.union[T.table, T.struct]):
    elems = t.elem_dict
    if t.nullable:
        elems = {k:v.replace(nullable=True) for k, v in elems.items()}
    return concat_for(flatten_path(path + [name], col) for name, col in elems.items())
@dp_type
def flatten_path(path, t: T.list):
    return concat_for(flatten_path(path + [name], col) for name, col in [('value', t.elem)])


def flatten_type(tp, path = []):
    # return [(join_names(path), t.col_type) for path, t in flatten_path([name], tp)]
    return [(join_names(path), t) for path, t in flatten_path(path, tp)]


def table_params(t):
    # TODO hide_from_init / writeable, not id
    return [(name, c) for name, c in t.elems.items() if not c <= T.t_id]



def table_flat_for_insert(table):
    # auto_count = join_names(self.primary_keys)
    # if 'pk' not in table.options:
    #     raise exc.pql_TypeError([], f"Cannot add to table. Primary key not defined")

    pks = {join_names(pk) for pk in table.options.get('pk', [])}
    names = [name for name,t in flatten_type(table)]
    return classify_bool(names, lambda name: name in pks)


@dp_inst
def repr_value(v: T.object):
    return repr(v.value)

@dp_inst
def repr_value(v: T.decimal):
    raise exc.pql_NotImplementedError([], "Decimal not implemented")

@dp_inst
def repr_value(v: T.string):
    return f'"{v.value}"'

@dp_inst
def repr_value(v: T.text):
    return str(v.value)

@dp_inst
def repr_value(v: T.bool):
    return 'true' if v.value else 'false'




@dp_inst
def from_sql(state, res: T.primitive):
    try:
        row ,= res.value
        item ,= row
    except ValueError:
        raise exc.pql_TypeError.make(state, None, "Expected primitive. Got: '%s'" % res.value)
    return item

def _from_datetime(s):
    if s is None:
        return None

    # Postgres
    if isinstance(s, datetime):
        return s

    # Sqlite
    if not isinstance(s, str):
        raise exc.pql_TypeError([], f"datetime expected a string. Instead got: {s}")
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise exc.pql_ValueError([], str(e))

@dp_inst
def from_sql(state, res: T.datetime):
    # XXX doesn't belong here?
    row ,= res.value
    item ,= row
    s = item
    return _from_datetime(s)

@dp_inst
def from_sql(state, arr: T.list):
    if not all(len(e)==1 for e in arr.value):
        raise exc.pql_TypeError(state, None, f"Expected 1 column. Got {len(arr.value[0])}")
    return [e[0] for e in arr.value]

@dp_inst
@listgen
def from_sql(state, arr: T.table):
    target = state.db.target
    expected_length = len(flatten_type(arr.type))   # TODO optimize?
    for row in arr.value:
        if len(row) != expected_length:
            raise exc.pql_TypeError.make(state, None, f"Expected {expected_length} columns, but got {len(row)}")
        i = iter(row)
        yield {name: restructure_result(target, col, i) for name, col in arr.type.elems.items()}

@dp_type
def restructure_result(target, t: T.table, i):
    # return ({name: restructure_result(target, col, i) for name, col in t.elem_dict.items()})
    return next(i)

@dp_type
def restructure_result(target, t: T.struct, i):
    return ({name: restructure_result(target, col, i) for name, col in t.elem_dict.items()})

@dp_type
def restructure_result(target, t: T.union[T.primitive, T.null], i):
    return next(i)


@dp_type
def restructure_result(target, t: T.list[T.primitive, T.null], i):
    # XXX specific to choice of db. So belongs in sql.py?
    res = next(i)
    if target == 'mysql':   # TODO use constant
        res = json.loads(res)
    elif target == 'sqlite':
        res = res.split('|')
    if t.elem <= T.int: # XXX hack! TODO Use a generic form
        res = [int(x) for x in res]
    elif t.elem <= T.float: # XXX hack! TODO Use a generic form
        res = [float(x) for x in res]
    return res

@dp_type
def restructure_result(target, t: T.datetime, i):
    s = next(i)
    return _from_datetime(s)


def join_names(names):
    return "_".join(names)
