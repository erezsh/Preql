import runtype
from runtype.typesystem import TypeSystem

from typing import Union
from datetime import datetime
from dataclasses import field
from decimal import Decimal

from .base import Object
from .utils import dataclass

global_methods = {}

class Id:
    def __init__(self, *parts):
        assert all(isinstance(p, str) for p in parts), parts
        self.parts = parts
    def __str__(self):
        raise Exception("Operation not allowed!")
    def __hash__(self):
        raise Exception("Operation not allowed!")
    @property
    def repr_name(self):
        return self.parts[-1]


def _repr_type_elem(t, depth):
    return _repr_type(t, depth-1) if isinstance(t, Type) else repr(t)

def _repr_type(t, depth=2):
    if t.elems:
        if depth > 0:
            if isinstance(t.elems, dict):
                elems = '[%s]' % ', '.join(f'{k}: {_repr_type_elem(v, depth)}' for k,v in t.elems.items())
            else:
                elems = '[%s]' % ', '.join(_repr_type_elem(e, depth) for e in t.elems)
        else:
            elems = '[...]'
    else:
        elems = ''
    return f'{t._typename_with_q}{elems}'

ITEM_NAME = 'item'

@dataclass
class Type(Object):
    typename: str
    supertypes: frozenset
    elems: Union[tuple, dict] = field(hash=False, default_factory=dict)
    options: dict = field(hash=False, compare=False, default_factory=dict)
    methods: dict = field(hash=False, compare=False, default_factory=lambda: dict(global_methods))
    _nullable: bool = field(default_factory=bool)

    @property
    def _typename_with_q(self):
        n = '?' if self._nullable else ''
        return f'{self.typename}{n}'

    @property
    def elem(self):
        if isinstance(self.elems, dict):
            elem ,= self.elems.values()
        else:
            elem ,= self.elems
        return elem

    def as_nullable(self):
        assert not self.maybe_null()
        return self.replace(_nullable=True)

    def maybe_null(self):
        return self._nullable or self is T.nulltype

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

        l1 = self.elems if isinstance(self.elems, dict) else dict(enumerate(self.elems))
        l2 = other.elems if isinstance(other.elems, dict) else dict(enumerate(other.elems))
        if len(l1) != len(l2):
            return False

        res = self.typename == other.typename and all(
            k1==k2 and v1.__eq__(v2, memo)
            for (k1,v1), (k2,v2) in zip(l1.items(), l2.items())
        )
        return res


    @property
    def elem_types(self):
        if isinstance(self.elems, dict):
            return self.elems.values()
        return self.elems

    def issubtype(self, t):
        assert isinstance(t, Type), t
        if self.typename == 'union':
            return all(t2.issubtype(t) for t2 in self.elem_types)
        elif t.typename == 'union':   # XXX a little hacky. Change to issupertype?
            return any(self.issubtype(t2) for t2 in t.elem_types)

        if self is T.nulltype:
            if t.maybe_null():
                return True

        # TODO zip should be aware of lengths
        if t.typename in {s.typename for s in self.supertype_chain()}:
            return all(e1.issubtype(e2) for e1, e2 in zip(self.elem_types, t.elem_types))
        return False

    def __le__(self, other):
        return self.issubtype(other)

    def __getitem__(self, elems):
        if self is T.union or self is T.function:
            elems = tuple(elems)
        else:
            assert not isinstance(elems, tuple), (self, elems)
            elems = {ITEM_NAME: elems}
        return self.replace(elems=elems)

    def __call__(self, elems=None, **options):
        return self.replace(elems=elems or self.elems, methods=dict(self.methods), options={**self.options, **options})

    def __repr__(self):
        # TODO Move to dp_inst?
        return _repr_type(self)

    def get_attr(self, attr):
        if self is T.unknown:
            return self

        # XXX hacky
        if attr == 'elem' and self.elems and len(self.elems) == 1:
            try:
                return self.elem
            except ValueError:
                pass
        # elif attr == 'elems' and self.elems:
        #     return self.elems

        assert attr not in self.methods

        return super().get_attr(attr)

    def all_attrs(self):
        if len(self.elems) == 1:
            return {'elem': self.elem}
        # else:
        #     return {'elems': self.elems}
        return {}

    def repr(self, state):
        return repr(self)



class TypeDict(dict):

    def _register(self, name, supertypes=(), elems=()):
        t = Type(name, frozenset(supertypes), elems)
        assert name not in self
        T[name] = t
        dict.__setattr__(self, name, t)

    def __setattr__(self, name, args):
        if isinstance(args, tuple):
            self._register(name, *args)
        else:
            self._register(name, args)



T = TypeDict()

T.any = ()
T.unknown = [T.any]

T.union = [T.any]
T.type = [T.any]
Type.type = T.type

T.object = [T.any]
T.nulltype = [T.object]

T.primitive = [T.object]

T.text = [T.primitive]
T.string = [T.text]
T.number = [T.primitive]
T.int = [T.number]
T.float = [T.number]
T.bool = [T.primitive]    # number
T.decimal = [T.number]

T.datetime = [T.primitive]    # primitive? struct?

T.container = [T.object]

T.struct = [T.container]
T.row = [T.struct]

T.collection = [T.container], {}
T.table = [T.collection], {}
T.list = [T.table], (T.any,)
T.set = [T.table], (T.any,)
T.aggregate = [T.collection], (T.any,)
T.t_id = [T.number], (T.table,)
T.t_relation = [T.number], (T.table,)   # t_id?

T.vectorized = [T.container], (T.any,)  # sequence or collection?

T.function = [T.object]
T.module = [T.object]

T.signal = [T.object]
#-----------

T.Exception = [T.signal]

T.IOError = [T.Exception]
T.CodeError = [T.Exception]
T.EvalError = [T.Exception]

# CodeError - Failures due to inherently unexecutable code
T.SyntaxError = [T.CodeError]
T.NotImplementedError = [T.CodeError]

# IOError - All errors resulting directly from attempts at I/O communication
T.FileError = [T.IOError]
T.DbError = [T.IOError]
T.DbQueryError = [T.DbError]
T.DbConnectionError = [T.DbError]

# EvalError - Errors that arise only when evaluating the code (either at run-time or compile-time)
T.TypeError = [T.EvalError]
T.ValueError = [T.EvalError]
T.NameError = [T.EvalError]
T.JoinError = [T.EvalError]
T.CompileError = [T.EvalError]

T.AttributeError = [T.NameError]
T.AssertError = [T.ValueError]
T.IndexError = [T.ValueError]
T.CastError = [T.TypeError]

T.ImportError = [T.Exception]   # XXX


#-------------


_t = {
    bool: T.bool,
    int: T.int,
    float: T.float,
    str: T.string,
    datetime: T.datetime,
    Decimal: T.decimal,
}
def from_python(t):
    # TODO throw proper exception if this fails
    return _t[t]


from collections import deque
def common_type(t1, t2):
    "Returns a type which is the closest ancestor of both t1 and t2"
    v1 = {t1}
    v2 = {t2}

    o1 = deque([t1])
    o2 = deque([t2])
    while o1 or o2:
        x1 = o1.popleft()
        v1.add(x1)
        if x1 in v2:
            return x1
        o1 += [t for t in x1.supertypes if t not in v1]

        x2 = o2.popleft()
        v2.add(x2)
        if x2 in v1:
            return x2
        o2 += [t for t in x2.supertypes if t not in v2]

    assert False


def union_types(types):
    # TODO flatten unions, remove duplications and subtypes
    ts = set(types)
    if len(ts) > 1:
        elem_type = T.union(elems=tuple(ts))
    else:
        elem_type ,= ts
    return elem_type



class ProtoTS(TypeSystem):
    def issubclass(self, t1, t2):
        if t2 is object:
            return True
        is_t2 = isinstance(t2, Type)
        if isinstance(t1, Type):
            return is_t2 and t1 <= t2
        elif is_t2:
            return False

        # Regular Python types
        return runtype.issubclass(t1, t2)

    default_type = object

class TS_Preql(ProtoTS):
    def get_type(self, obj):
        try:
            return obj.type
        except AttributeError:
            return type(obj)



class TS_Preql_subclass(ProtoTS):
    def get_type(self, obj):
        # Preql objects
        if isinstance(obj, Type):
            return obj

        # Regular Python
        return type(obj)

dp_type = runtype.Dispatch(TS_Preql_subclass())
dp_inst = runtype.Dispatch(TS_Preql())
