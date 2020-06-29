from typing import List, Tuple, Optional, Union
from dataclasses import field
from datetime import datetime

import runtype
from runtype.typesystem import TypeSystem

from typing import List, Dict, Optional, Any, Union
from datetime import datetime

from .utils import dataclass, listgen, concat_for, SafeDict, classify_bool, safezip
from . import exceptions as exc
from dataclasses import field, replace

@dataclass
class Object:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"
    # dyn_attrs: dict = field(default_factory=dict, init=False, compare=False)

    def repr(self, pql):
        return repr(self)

    def get_attr(self, attr):
        raise exc.pql_AttributeError([], f"{self} has no attribute: {attr}")    # XXX TODO

    def isa(self, t):
        if not isinstance(t, Type):
            raise exc.pql_TypeError([], f"'type' argument to isa() isn't a type. It is {t}")
        return self.type <= t


class AbsType:
    pass

global_methods = {}

@dataclass
class Type(Object, AbsType):
    typename: str
    supertypes: frozenset
    elems: Union[tuple, dict] = field(hash=False, default_factory=dict)
    options: dict = field(hash=False, compare=False, default_factory=dict)
    methods: dict = field(hash=False, compare=False, default_factory=lambda: dict(global_methods))
    nullable: bool = field(default_factory=bool)

    @property
    def typename_with_q(self):
        n = '?' if self.nullable else ''
        return f'{self.typename}{n}'

    @property
    def elem(self):
        if isinstance(self.elems, dict):
            elem ,= self.elems.values()
        else:
            elem ,= self.elems
        return elem

    def supertype_chain(self):
        res = {
            t2
            for t1 in self.supertypes
            for t2 in t1.supertype_chain()
        }

        assert self not in res
        return res | {self}

    def __eq__(self, other, memo=None):
        "Repetitive nested equalities are assumed to be true"

        if not isinstance(other, Type):
            return False

        if memo is None:
            memo = set()

        a, b = id(self), id(other)
        if (a,b) in memo or (b,a) in memo:
            return True

        memo.add((a, b))

        l1 = self.elems if isinstance(self.elems, tuple) else list(self.elems.values())
        l2 = other.elems if isinstance(other.elems, tuple) else list(other.elems.values())
        if len(l1) != len(l2):
            return False

        return self.typename == other.typename and all(
            i1.__eq__(i2, memo) for i1, i2 in zip(l1, l2)
        )


    @property
    def elem_types(self):
        if isinstance(self.elems, dict):
            return self.elems.values()
        return self.elems

    def issubtype(self, t):
        assert isinstance(t, Type), t
        if t.typename == 'union':   # XXX a little hacky. Change to issupertype?
            return any(self.issubtype(t2) for t2 in t.elem_types)

        # TODO zip should be aware of lengths
        if t.typename in {s.typename for s in self.supertype_chain()}:
            return all(e1.issubtype(e2) for e1, e2 in zip(self.elem_types, t.elem_types))
        return False

    def __le__(self, other):
        return self.issubtype(other)

    def __getitem__(self, elems):
        if not isinstance(elems, tuple):
            elems = elems,
        return self.replace(elems=tuple(elems))

    def __call__(self, **kw):
        return self.replace(elems=kw, methods=dict(self.methods))

    def set_options(self, **kw):
        options = dict(self.options)
        options.update(kw)
        return self.replace(options=options)

    def __repr__(self):
        # TODO fix. Move to pql_dp?
        if self.elems:
            if isinstance(self.elems, dict):
                elems = '[%s]' % ', '.join(f'{k}: {v.typename_with_q if isinstance(v, Type) else repr(v)}' for k,v in self.elems.items())
            else:
                elems = '[%s]' % ', '.join(e.typename if isinstance(e, Type) else repr(e) for e in self.elems)
        else:
            elems = ''
        return f'{self.typename_with_q}{elems}'

    def get_attr(self, attr):
        if self is T.unknown:
            return self

        # XXX hacky
        if attr == 'elem' and self.elems:
            try:
                return self.elem
            except ValueError:
                pass

        assert attr not in self.methods

        return super().get_attr(attr)



class TypeDict(dict):
    def __getattr__(self, t):
        return self[t]

    def _register(self, name, supertypes=(), elems=()):
        t = Type(name, frozenset(supertypes), elems)
        assert name not in T
        T[name] = t
        dict.__setattr__(T, name, t)

    def __setattr__(self, name, args):
        self._register(name, *args)



T = TypeDict()

T.any = ()
T.unknown = [T.any],

T.union = [T.any],
T.type = [T.any],
Type.type = T.type

T.object = [T.any],
T.null = [T.object],

T.primitive = [T.object],

T.text = [T.primitive],
T.string = [T.text],
T.number = [T.primitive],
T.int = [T.number],
T.float = [T.number],
T.bool = [T.primitive],    # number?
T.decimal = [T.number],

T.datetime = [T.primitive],    # primitive? struct?

T.container = [T.object], #(T.object,)

T.struct = [T.container], {}
T.row = [T.struct], {}

T.collection = [T.container],
T.table = [T.collection], {}
T.list = [T.table], (T.object,)
T.set = [T.table], (T.object,)
T.aggregate = [T.collection], (T.object,)
T.t_id = [T.number], (T.table,)
T.t_relation = [T.number], (T.table,)   # t_id?

T.function = [T.object], {}

T.exception = [T.object], {}
#-----------

def join_names(names):
    return "_".join(names)


from decimal import Decimal
_t = {
    bool: T.bool,
    int: T.int,
    float: T.float,
    str: T.string,
    datetime: T.datetime,
    Decimal: T.decimal,
}
def from_python(t):
    return _t[t]


#---------------------

class MyTypeSystem(TypeSystem):
        # Preql objects
    # def issubclass(self, t1, t2):
    #     if isinstance(t2, Type)
    #     return t1.issubtype(t2)
    def issubclass(self, t1, t2):
        if t2 is object:
            return True
        is_t2 = isinstance(t2, Type)
        if isinstance(t1, Type):
            return is_t2 and t1 <= t2
        elif is_t2:
            return False

        # Regular Python
        return runtype.issubclass(t1, t2)

    def get_type(self, obj):
        try:
            return obj.type
        except AttributeError:
            return type(obj)

    def canonize_type(self, t):
        return t

    default_type = object #T.any

pql_dp = runtype.Dispatch(MyTypeSystem())


@pql_dp
def repr_value(v):
    return repr(v.value)

@pql_dp
def repr_value(v: T.decimal):
    raise exc.pql_NotImplementedError([], "Decimal not implemented")

@pql_dp
def repr_value(v: T.string):
    return f'"{v.value}"'

@pql_dp
def repr_value(v: T.text):
    return str(v.value)

@pql_dp
def repr_value(v: T.bool):
    return 'true' if v.value else 'false'


@pql_dp
def from_sql(state, res: T.primitive):
    row ,= res.value
    item ,= row
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

@pql_dp
def from_sql(state, res: T.datetime):
    # XXX doesn't belong here?
    row ,= res.value
    item ,= row
    s = item
    return _from_datetime(s)

@pql_dp
def from_sql(state, arr: T.list):
    # TODO not assert
    if not all(len(e)==1 for e in arr.value):
        raise exc.pql_TypeError(state, None, f"Expected 1 column. Got {len(arr.value[0])}")
    return [e[0] for e in arr.value]

@pql_dp
@listgen
def from_sql(state, arr: T.table):
    expected_length = len(flatten_type(arr.type))   # TODO optimize?
    for row in arr.value:
        if len(row) != expected_length:
            raise exc.pql_TypeError.make(state, None, f"Expected {expected_length} column. Got {len(row)}")
        i = iter(row)
        yield {name: restructure_result(col, i) for name, col in arr.type.elems.items()}






class TypeSystem2(MyTypeSystem):
        # Preql objects
    def issubclass(self, t1, t2):
        if t2 is object:
            return True
        is_t2 = isinstance(t2, Type)
        if isinstance(t1, Type):
            return is_t2 and t1 <= t2
        elif is_t2:
            return False

        # Regular Python
        return runtype.issubclass(t1, t2)

    def get_type(self, obj):
        # Preql objects
        if isinstance(obj, Type):
            return obj

        # Regular Python
        return type(obj)

    default_type = object


combined_dp = runtype.Dispatch(TypeSystem2())

@combined_dp
def flatten_path(path, t):
    return [(path, t)]

@combined_dp
def flatten_path(path, t: T.union[T.table, T.struct]):
    elems = table_to_struct(t).elems
    if t.nullable:
        elems = {k:v.replace(nullable=True) for k, v in elems.items()}
    return concat_for(flatten_path(path + [name], col) for name, col in elems.items())
@combined_dp
def flatten_path(path, t: T.list):
    return concat_for(flatten_path(path + [name], col) for name, col in [('value', t.elem)])


def flatten_type(tp, path = []):
    # return [(join_names(path), t.col_type) for path, t in flatten_path([name], tp)]
    return [(join_names(path), t) for path, t in flatten_path(path, tp)]


def table_params(t):
    # TODO hide_from_init, not id
    return [(name, c) for name, c in t.elems.items() if not c <= T.t_id]


@combined_dp
def restructure_result(t: T.struct, i):
    return ({name: restructure_result(col, i) for name, col in table_to_struct(t).elems.items()})

@combined_dp
def restructure_result(t: T.union[T.primitive, T.null], i):
    return next(i)

@combined_dp
def restructure_result(t: T.datetime, i):
    s = next(i)
    return _from_datetime(s)


def table_flat_for_insert(table):
    # auto_count = join_names(self.primary_keys)
    # if 'pk' not in table.options:
    #     raise exc.pql_TypeError([], f"Cannot add to table. Primary key not defined")

    pks = {join_names(pk) for pk in table.options.get('pk', [])}
    names = [name for name,t in flatten_type(table)]
    return classify_bool(names, lambda name: name in pks)

def table_to_struct(t):
    "Misnamed.."
    elems = t.elems
    if isinstance(elems, tuple):
        assert len(elems) == 1
        elems = {'value': elems[0]}
    return T.struct(**elems)


global_methods['zz'] = T.int