from typing import List, Dict, Optional, Any

from .utils import dataclass, listgen, concat_for, SafeDict

class PqlObject:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"

class PqlType(PqlObject):
    """PqlType annotates the type of all instances """

    def concrete_type(self):
        return self
    def kernel_type(self):
        return self


# Primitives
class NullType(PqlType):
    def import_result(self, res):
        return None

null = NullType()

@dataclass
class Primitive(PqlType):
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

# TODO nullable!!!
class Text(str):
    pass



# class Date(Primitive): pass
# class Text(Primitive): pass
# class Json(Primitive): pass
from datetime import datetime

Int = Primitive('int', int, False)
Float = Primitive('float', float, False)
String = Primitive('string', str, False)
Text = Primitive('text', Text, False)
Bool = Primitive('bool', bool, False)
Date = Primitive('date', datetime, False)   # XXX datetime?

# Collections

class Collection(PqlType):

    def to_struct_type(self):
        return StructType(self.name, {name: col.type for name, col in self.columns.items()})

@dataclass
class ListType(Collection):
    elemtype: PqlType

    @property
    def columns(self):
        return {'value': make_column(self.elemtype)}
    @property
    def name(self):
        return 'list_%s' % self.elemtype.name

    def kernel_type(self):
        return self.elemtype

    def import_result(self, arr):
        assert all(len(e)==1 for e in arr)
        return [e[0] for e in arr]

    def restructure_result(self, res):
        # return next(res)
        x = next(res)
        return x

    def flat_length(self):
        return 1

    def flatten(self, path):
        return [(path, self)]

    def __repr__(self):
        return f'list[{self.elemtype}]'

class Aggregated(ListType):
    pass

@dataclass
class SetType(Collection):
    elemtype: PqlType


class ColumnType(PqlType):
    def flatten(self, path):
        return [(path, self)]

    def concrete_type(self):
        return self.type

    is_concrete = True  # Concrete = actually contains data
    primary_key = False
    readonly = False

    default = None  # TODO


@dataclass
class TableType(Collection):
    name: str
    columns: Dict[str, ColumnType]
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
        # return f'TableType({self.name}, [{", ".join(list(self.columns))}])'
        return f'TableType({self.name})'

    def _data_columns(self):
        return [c.type if not isinstance(c.type, IdType) else "id" for c in self.columns.values()]

    def __hash__(self):
        # raise NotImplementedError()
        # return hash((self.name, tuple(self.columns.items())))
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
            s = ({str(name): col.type.restructure_result(i) for name, col in self.columns.items()})
            yield s



@dataclass
class StructType(Collection):
    name: str
    members: Dict[str, PqlType]

    def restructure_result(self, i):
        return ({name: col.restructure_result(i) for name, col in self.members.items()})

    def __hash__(self):
        # XXX Do members really don't matter? Isn't there useful info there?
        members = tuple(self.members.items())
        return hash((self.name, members))

    def __repr__(self):
        return f'<struct {self.name}{tuple(self.members.values())}>'


@dataclass
class DatumColumnType(ColumnType):
    type: PqlType
    primary_key: bool = False
    readonly: bool = False

    def restructure_result(self, i):
        return self.type.restructure_result(i)

@dataclass
class StructColumnType(ColumnType):
    type: (StructType, Aggregated)
    members: Dict[str, ColumnType]

    def flatten(self, path):
        return concat_for(col.flatten(path + [name]) for name, col in self.members.items())

    def __hash__(self):
        return hash(self.type)  # XXX Do members really don't matter? Isn't there useful info there?

@dataclass
class RelationalColumnType(ColumnType):
    type: PqlType
    query: Optional[Any] = None # XXX what now?

def make_column(type_, query=None):
    kernel = type_.kernel_type()
    if isinstance(kernel, StructType):
        assert not query
        return StructColumnType(type_, {
            n: make_column(m) for (n,m) in kernel.members.items()
        })
    elif query or isinstance(kernel, TableType):
        return RelationalColumnType(type_.concrete_type(), query)
    else:
        return DatumColumnType(type_)
    assert False, type_


@dataclass
class IdType(PqlType):
    table: TableType

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
