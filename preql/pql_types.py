from typing import List, Dict, Optional, Any, Union
from datetime import datetime

from .utils import dataclass, listgen, concat_for, SafeDict, classify_bool
from . import exceptions as exc
from dataclasses import field, replace

@dataclass
class PqlObject:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"
    dyn_attrs: dict = field(default_factory=dict, init=False, compare=False)

    def repr(self, pql):
        return repr(self)

    def get_attr(self, attr):
        raise exc.pql_AttributeError(attr.meta, f"{self} has no attribute: {attr}")

    # def is_equal(self, other):
    #     raise exc.pql_NotImplementedError(f"Equality of {self} not implemented")
    def isa(self, t):
        if not isinstance(t, PqlType):
            raise exc.pql_TypeError(None, f"'type' argument to isa() isn't a type. It is {t}")
        return self.type.issubclass(t)

    def replace(self, **attrs):
        if 'dyn_attrs' in attrs:
            assert not attrs.pop('dyn_attrs')
        return replace(self, **attrs)



class PqlType(PqlObject):
    """PqlType annotates the type of all instances """

    @property
    def col_type(self):
        return self

    @property
    def code(self):
        raise exc.pql_TypeError(None, "type objects are local and cannot be used in target code.")

    def kernel_type(self):
        return self

    def composed_of(self, t):
        return isinstance(self, t)

    def effective_type(self):   # XXX what
        return self

    def flatten_type(self, path_base=[]):
        return [(join_names(path), t.col_type) for path, t in self.flatten_path(path_base)]

    def flatten_path(self, *args):
        raise NotImplementedError(self)

    def apply_inner_type(self, t):
        raise TypeError("This type isn't a 'generic', and has no inner type")

    def __repr__(self):
        assert type(self) is PqlType, self
        return 'type'

    def repr_value(self, value):
        return repr(value)

    def issubclass(self, t):
        # XXX this is incorrect. issubclass(int, type) returns true, when it shouldn't
        # XXX sublassing should probably be explicit, not through python
        assert isinstance(t, PqlType)
        return isinstance(self, type(t))
        # return self == t

    hide_from_init = False

PqlType.type = PqlType()

# Primitives

class AnyType(PqlType):
    name = 'any'

    def __repr__(self):
        return self.name

    def flatten_path(self, path):
        # XXX Only works because for now it's only used for list
        # TODO better type system
        return [(path, self)]



class AtomicOrList(PqlType):
    "For types that are the width of one column. Basically non-table?"

    def flatten_path(self, path):
        return [(path, self)]

    def restructure_result(self, res):
        return next(res)


class AtomicType(AtomicOrList):
    # XXX these don't belong here!
    default = None


class NullType(AtomicType):
    name = 'null'
    # def import_result(self, res):
    #     assert res is None
    #     return None

    def __repr__(self):
        return 'null'

    def repr_value(self, value):
        assert value is None, value
        return repr(self)


class Primitive(AtomicType):
    pytype: type

    by_pytype = {}  # Singleton

    @property
    def name(self):
        return type(self).__name__.lstrip('_').lower()

    def __repr__(self):
        return self.name

    def __init__(self):
        assert self.pytype not in self.by_pytype
        self.by_pytype[self.pytype] = self

    def import_result(self, res):
        row ,= res
        item ,= row
        return item


class text(str):
    pass

class _DateTime(Primitive):
    pytype = datetime

    def import_result(self, res):
        s = super().import_result(res)
        if s:
            if not isinstance(s, str):
                raise exc.pql_TypeError(None, f"Expected a string. Instead got: {s}")
            try:
                return datetime.fromisoformat(s)
            except ValueError as e:
                raise exc.pql_ValueError(None, str(e))

    def restructure_result(self, i):
        s = super().restructure_result(i)
        return datetime.fromisoformat(s)


class Number(Primitive):
    pass

class _Int(Number):  pytype = int
class _Float(Number):  pytype = float
class _String(Primitive):
    pytype = str

    def repr_value(self, value):
        return f'"{value}"'

class _Text(Primitive):
    pytype = text

    def repr_value(self, value):
        return str(value)

class _Bool(Primitive):  pytype = bool

Int = _Int()
Float = _Float()
String = _String()
Text = _Text()
Bool = _Bool()
DateTime = _DateTime()
null = NullType()
any_t = AnyType()
object_t = PqlObject()


# Collections

class Collection(PqlType):

    def to_struct_type(self):
        return StructType(self.name, {name: col.col_type for name, col in self.columns.items()})

    def flatten_path(self, path=[]):
        return concat_for(col.flatten_path(path + [name]) for name, col in self.columns.items())

    def column_codename(self, name):
        if self.codenames is None:
            return name
        return self.codenames[name]

    def columns_with_codenames(self):
        return [(n,t,self.column_codename(n)) for n,t in self.columns.items()]

    def get_attr(self, name):
        try:
            return self.columns[name].col_type
        except KeyError:
            try:
                return self.dyn_attrs[name]
            except KeyError:
                raise exc.pql_AttributeError(None, name)



@dataclass
class ListType(Collection, AtomicOrList):
    elemtype: PqlType

    codenames: object = None

    primary_keys = []

    @property
    def columns(self):
        return SafeDict({'value': self.elemtype})
    @property
    def name(self):
        return 'list_%s' % self.elemtype.name

    def import_result(self, arr):
        assert all(len(e)==1 for e in arr)
        return [e[0] for e in arr]

    def flat_length(self):
        return 1

    def __repr__(self):
        return f'list[{self.elemtype}]'

    def apply_inner_type(self, t):
        return type(self)(t)


@dataclass
class FunctionType(PqlType):
    param_types: List[Any]
    param_collector: bool

    name = "function"

    def __repr__(self):
        types_str = ', '.join(repr(t) for t in self.param_types)
        return f'function({types_str})'


@dataclass
class OptionalType(PqlType):
    type: PqlType

    def flatten_path(self, path):
        return [(p, OptionalType(t)) for p, t in self.type.flatten_path(path)]

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def kernel_type(self):
        return self.type.kernel_type()

    def composed_of(self, t):
        return self.type.composed_of(t)

    @property
    def members(self):  # XXX this is just a hack
        return {n:OptionalType(t) for n,t in self.type.members.items()}

    def get_column_type(self, name):
        return OptionalType( self.type.get_column_type(name) )

# Not supported by Postgres. Will require a trick (either alternating within a column tuple, or json type, etc)
# @dataclass
# class UnionType(AtomicType):
#     types: List[AtomicType]

#     @property
#     def name(self):
#         return 'union_%s' % join_names(t.name for t in self.types)

#     def __repr__(self):
#         return f'union{self.types}'

#     def __hash__(self):
#         return hash(tuple(self.types))


@dataclass
class Aggregated(AtomicOrList):
    elemtype: PqlType


@dataclass
class SetType(Collection):
    elemtype: PqlType


class Column:
    def repr(self, pql):
        return self.type.repr(pql)

@dataclass
class TableType(Collection):
    name: str
    columns: Dict[str, Union[PqlType, Column]]
    temporary: bool
    primary_keys: List[List[str]]
    autocount: List[str] = field(default_factory=list)

    codenames: object = None

    def flatten_path(self, path=[]):
        return concat_for(col.flatten_path(path + [self.column_codename(name)]) for name, col in self.columns.items())

    def __post_init__(self):
        # super().__post_init__()
        assert isinstance(self.columns, SafeDict), self.columns

    def flat_for_insert(self):
        auto_count = join_names(self.autocount)
        names = [name for name,t in self.flatten_type()]
        return classify_bool(names, lambda name: name==auto_count)

    def params(self):
        return [(name, c) for name, c in self.columns.items() if not c.hide_from_init]

    def get_column_type(self, name):
        return self.columns[name].col_type

    def flat_length(self):
        # Maybe memoize
        return len(self.flatten_path())

    def __repr__(self):
        return f'TableType({self.name})'
        # return f'TableType({self.name}, {{{", ".join(repr(t) for t in self.columns.values())}}})'

    def repr(self, pql):
        return f'{self.name}{{{", ".join(t.repr(pql) for t in self.columns.values())}}}'

    def _data_columns(self):
        # XXX idtypes don't care about name?
        return [(name, c) if not isinstance(c, IdType) else "id" for name, c in self.columns.items()]

    def __hash__(self):
        return hash(tuple(self._data_columns()))

    def __eq__(self, other):
        return self._data_columns() == other._data_columns()

    def restructure_result(self, i):
        "Called from import_result, to read the table's id (not actual structure)"
        return next(i)

    @listgen
    def import_result(self, arr):
        expected_length = self.flat_length()
        for row in arr:
            assert len(row) == expected_length, (expected_length, row)
            i = iter(row)
            s = ({str(name): col.restructure_result(i) for name, col in self.columns.items()})
            yield s

@dataclass
class RelationalColumn(Column):
    type: PqlType
    query: Optional[Any] = None # XXX what now?

    def __post_init__(self):
        assert self.type.composed_of(TableType)

    def flatten_path(self, path):
        return [(path, self)]

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def effective_type(self):   # XXX Yikes
        return self.get_pk()

    def get_pk(self):   # XXX Yikes
        t = self.type.kernel_type()
        pks = [join_names(pk) for pk in t.primary_keys]
        assert len(pks) == 1
        return self.type.get_column_type(pks[0])    # TODO what if there's more than one? Struct relation.. ??

    def __hash__(self):
        return id(self.type)    # TODO not good! XXX Hack to avoid infinite recursion

    @property
    def col_type(self):
        return self.get_pk().col_type

    hide_from_init = False
    default = None

@dataclass
class DataColumn(Column):
    type: PqlType
    default: Optional[PqlObject] = None

    @property
    def col_type(self):
        return self.type

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def flatten_path(self, path):
        return self.type.flatten_path(path)

    @property
    def hide_from_init(self):
        return self.type.hide_from_init



@dataclass
class StructType(PqlType):
    name: str
    members: Dict[str, PqlType]

    default = None

    def restructure_result(self, i):
        return ({name: col.restructure_result(i) for name, col in self.members.items()})

    def __hash__(self):
        # XXX only compare types?
        members = tuple(self.members.items())
        return hash((self.name, members))

    def __repr__(self):
        return f'<struct {self.name}{tuple(self.members.values())}>'

    def flatten_path(self, path):
        return concat_for(col.flatten_path(path + [name]) for name, col in self.members.items())

    def columns_with_codenames(self):
        return [(n,t,n) for n,t in self.members.items()]

    @property
    def columns(self):
        # XXX time to give it up?
        return self.members

@dataclass
class IdType(_Int):
    table: TableType

    hide_from_init = True

    def __repr__(self):
        return f'{self.table.name}.id'

def join_names(names):
    return "_".join(names)

@dataclass
class RowType(PqlType):
    table: Collection
    # idtype: IdType

    def import_result(self, arr):
        r ,= self.table.import_result(arr)
        return r

