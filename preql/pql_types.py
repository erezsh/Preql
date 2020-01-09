from typing import List, Dict, Optional, Any
from datetime import datetime

from .utils import dataclass, listgen, concat_for, SafeDict

class PqlObject:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"

class PqlType(PqlObject):
    """PqlType annotates the type of all instances """

    def kernel_type(self):
        return self

    def actual_type(self):
        return self

    def repr(self, pql):
        return repr(self)

# Primitives
class NullType(PqlType):
    name = 'null'
    def import_result(self, res):
        return None

    def __repr__(self):
        return self.name

    def flatten(self, path):
        return [(path, self)]

    def restructure_result(self, res):
        return next(res)


null = NullType()


class AtomicType(PqlType):
    def flatten(self, path):
        return [(path, self)]

    def restructure_result(self, res):
        return next(res)

    # XXX these don't belong here!
    is_concrete = True
    primary_key = False
    readonly = False
    default = None


@dataclass
class Primitive(AtomicType):
    name: str
    pytype: type
    nullable: bool


    def __repr__(self):
        return self.name

    def __created__(self):
        primitives_by_pytype[self.pytype] = self

    def import_result(self, res):
        row ,= res
        item ,= row
        return item

    def restructure_result(self, i):
        return next(i)

    def repr(self, pql):
        return repr(self)

primitives_by_pytype = {}

class Text(str):
    pass

Int = Primitive('int', int, False)
Float = Primitive('float', float, False)
String = Primitive('string', str, False)
Text = Primitive('text', Text, False)
Bool = Primitive('bool', bool, False)
DateTime = Primitive('datetime', datetime, False)   # XXX datetime?

# Collections

class Collection(PqlType):

    def to_struct_type(self):
        return StructType(self.name, {name: col for name, col in self.columns.items()})

@dataclass
class ListType(Collection):
    elemtype: PqlType

    @property
    def columns(self):
        return {'value': self.elemtype}
    @property
    def name(self):
        return 'list_%s' % self.elemtype.name

    def kernel_type(self):
        return self.elemtype

    def import_result(self, arr):
        assert all(len(e)==1 for e in arr)
        return [e[0] for e in arr]

    def restructure_result(self, res):
        return next(res)

    def flat_length(self):
        return 1

    def flatten(self, path):
        return [(path, self)]

    def __repr__(self):
        return f'list[{self.elemtype}]'


# Not supported by Postgres. Will require a trick (either alternating columns, or json type, etc)
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
class DatumColumn(PqlType):
    type: PqlType
    default: Optional[PqlObject] = None

    def actual_type(self):
        return self.type

    def restructure_result(self, i):
        return self.type.restructure_result(i)

    def flatten(self, path):
        return self.type.flatten(path)

    @property
    def is_concrete(self):
        return self.type.is_concrete
    @property
    def readonly(self):
        return self.type.readonly

@dataclass
class RelationalColumn(AtomicType):
    type: PqlType
    query: Optional[Any] = None # XXX what now?

    def restructure_result(self, i):
        return self.type.restructure_result(i)

@dataclass
class TableType(Collection):
    name: str
    columns: Dict[str, PqlType]
    temporary: bool

    def __created__(self):
        assert isinstance(self.columns, SafeDict)

    def flatten(self, path=[]):
        return concat_for(col.flatten(path + [name]) for name, col in self.columns.items())

    def params(self):
        return [(name, c) for name, c in self.columns.items() if c.is_concrete and not c.readonly]

    def flat_length(self):
        # Maybe memoize
        return len(self.flatten())

    def __repr__(self):
        return f'TableType({self.name})'

    def repr(self, pql):
        # return repr(self)
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
class StructType(Collection):
    name: str
    members: Dict[str, PqlType]

    is_concrete = True
    readonly = False
    default = None

    def restructure_result(self, i):
        return ({name: col.restructure_result(i) for name, col in self.members.items()})

    def __hash__(self):
        # XXX only compare types?
        members = tuple(self.members.items())
        return hash((self.name, members))

    def __repr__(self):
        return f'<struct {self.name}{tuple(self.members.values())}>'

    def flatten(self, path):
        return concat_for(col.flatten(path + [name]) for name, col in self.members.items())


@dataclass
class IdType(AtomicType):
    table: TableType

    primary_key = True
    readonly = True

    def restructure_result(self, i):
        return next(i)

    def __repr__(self):
        return f'{self.table.name}.id'


@dataclass
class FunctionType(PqlType):
    param_types: List[Any]
    param_collector: bool

    name = "function"

    def __repr__(self):
        # types_str = ', '.join(repr(t) for t in self.param_types)
        return f'function(...)'

    def repr(self, pql):
        return repr(self)
