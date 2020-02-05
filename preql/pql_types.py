from typing import List, Dict, Optional, Any
from datetime import datetime

from .utils import dataclass, listgen, concat_for, SafeDict

class PqlObject:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"

    def repr(self, pql):
        return repr(self)

class PqlType(PqlObject):
    """PqlType annotates the type of all instances """

    def kernel_type(self):
        return self

    def actual_type(self):
        return self

    def effective_type(self):   # XXX what
        return self

    def flatten_type(self, path_base=[]):
        return [('_'.join(path), t) for path, t in self.flatten_path(path_base)]

    def apply_inner_type(self, t):
        raise TypeError("This type isn't a 'generic', and has no inner type")

    hide_from_init = False
    primary_key = False

# Primitives

class AnyType(PqlType):
    name = 'any'

    def __repr__(self):
        return self.name



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
    # def import_result(self, res):
    #     assert res is None
    #     return None

    def __repr__(self):
        return 'null'


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
        return datetime.fromisoformat(s)

    def restructure_result(self, i):
        s = super().restructure_result(i)
        return datetime.fromisoformat(s)


class Number(Primitive):
    pass

class _Int(Number):  pytype = int
class _Float(Number):  pytype = float
class _String(Primitive):  pytype = str
class _Text(Primitive):  pytype = text
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
        return StructType(self.name, {name: col for name, col in self.columns.items()})

@dataclass
class ListType(Collection, AtomicOrList):
    elemtype: PqlType

    primary_keys = []

    @property
    def columns(self):
        return SafeDict({'value': self.elemtype})
    @property
    def name(self):
        return 'list_%s' % self.elemtype.name

    def kernel_type(self):
        return self.elemtype

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
        # types_str = ', '.join(repr(t) for t in self.param_types)
        return f'function(...)'


@dataclass
class OptionalType(PqlType):
    type: PqlType

    def flatten_path(self, path):
        return [(p, OptionalType(t)) for p, t in self.type.flatten_path(path)]

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def kernel_type(self):
        return self.type.kernel_type()

# Not supported by Postgres. Will require a trick (either alternating within a column tuple, or json type, etc)
# @dataclass
# class UnionType(AtomicType):
#     types: List[AtomicType]

#     @property
#     def name(self):
#         return 'union_%s' % '_'.join(t.name for t in self.types)

#     def __repr__(self):
#         return f'union{self.types}'

#     def __hash__(self):
#         return hash(tuple(self.types))


class Aggregated(ListType):
    pass

@dataclass
class SetType(Collection):
    elemtype: PqlType



@dataclass
class TableType(Collection):
    name: str
    columns: Dict[str, PqlType]
    temporary: bool
    primary_keys: List[List[str]]

    def __post_init__(self):
        assert isinstance(self.columns, SafeDict), self.columns

    def flatten_path(self, path=[]):
        return concat_for(col.flatten_path(path + [name]) for name, col in self.columns.items())

    def flat_for_insert(self):
        columns = [name for name, _t in self.flatten_type()]
        primary_keys = {'_'.join(pk) for pk in self.primary_keys}   # XXX
        columns = [c for c in columns if c not in primary_keys]
        return list(primary_keys), columns

    def params(self):
        return [(name, c) for name, c in self.columns.items() if not c.hide_from_init]

    def flat_length(self):
        # Maybe memoize
        return len(self.flatten_path())

    def __repr__(self):
        # return f'TableType({self.name})'
        return f'TableType({self.name}, {{{", ".join(repr(t) for t in self.columns.values())}}})'

    def repr(self, pql):
        return f'{self.name}{{{", ".join(t.repr(pql) for t in self.columns.values())}}}'

    def _data_columns(self):
        return [c if not isinstance(c, IdType) else "id" for c in self.columns.values()]

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
class RelationalColumn(AtomicType):
    type: TableType
    query: Optional[Any] = None # XXX what now?

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def effective_type(self):   # XXX Yikes
        pks = ['_'.join(pk) for pk in self.type.primary_keys]
        return self.type.columns[pks[0]]    # TODO what if there's more than one? Struct relation.. ??

@dataclass
class DatumColumn(PqlType):
    type: PqlType
    default: Optional[PqlObject] = None

    def actual_type(self):
        return self.type

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def flatten_path(self, path):
        return self.type.flatten_path(path)

    @property
    def hide_from_init(self):
        return self.type.hide_from_init


@dataclass
class RowType(PqlType):
    table: Collection

    def import_result(self, arr):
        r ,= self.table.import_result(arr)
        return r


@dataclass
class StructType(Collection):
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


@dataclass
class IdType(_Int):
    table: TableType

    primary_key = True
    hide_from_init = True

    def __repr__(self):
        return f'{self.table.name}.id'

