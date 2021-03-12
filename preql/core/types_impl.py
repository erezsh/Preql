from .exceptions import Signal, pql_AttributeError

from preql.utils import concat_for, classify_bool

from .base import Object
from .pql_types import ITEM_NAME, T, Type, dp_type


def Object_get_attr(self, attr):
    raise pql_AttributeError(attr)

def Object_isa(self, t):
    if not isinstance(t, Type):
        raise Signal.make(T.TypeError, None, f"'type' argument to isa() isn't a type. It is {t}")
    return self.type <= t

Object.get_attr = Object_get_attr
Object.isa = Object_isa

def _type_flatten_code(self):
    raise Signal.make(T.TypeError, None, f"Found type 'type' in unexpected place")

Type.flatten_code = _type_flatten_code

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
    return [(join_names(path), t) for path, t in flatten_path(path, tp)]


def table_params(t):
    # TODO hide_from_init / writeable, not id
    return [(name, c) for name, c in t.elems.items() if not c <= T.t_id]



def table_flat_for_insert(table):
    pks = {join_names(pk) for pk in table.options.get('pk', [])}
    names = [name for name, t in flatten_type(table)]
    return classify_bool(names, lambda name: name in pks)


def join_names(names):
    return "_".join(names)


# The rest is implemented in display.py
@dp_type
def pql_repr(t, value):
    return repr(value)


def kernel_type(t):
    if t <= T.projected: # or t <= T.aggregated:
        return kernel_type(t.elems['item'])
    return t

