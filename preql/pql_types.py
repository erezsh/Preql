from typing import List, Dict, Optional, Any

from .utils import dataclass

class PqlObject:    # XXX should be in a base module
    "Any object that the user might interact with through the language, as so has to behave like an object inside Preql"

class PqlType(PqlObject):
    """PqlType annotates the type of all instances """


# Primitives
class NullType(PqlType): pass
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

primitives_by_pytype = {}

# TODO nullable!!!

# class Int(Primitive): pass
# class Float(Primitive): pass
# class String(Primitive): pass
# class Date(Primitive): pass
# class Bool(Primitive): pass
# class Text(Primitive): pass
# class Json(Primitive): pass
Int = Primitive('int', int, False)
Float = Primitive('float', float, False)
String = Primitive('string', str, False)
Bool = Primitive('bool', bool, False)
Date = Primitive('date', str, False)   # XXX datetime?

# Collections

class Collection(PqlType): pass

@dataclass
class ListType(Collection):
    elemtype: PqlType

    @property
    def columns(self):
        return {'value': make_column('value', self.elemtype)}
    @property
    def name(self):
        return 'list_%s' % self.elemtype.name

    def import_result(self, arr):
        assert all(len(e)==1 for e in arr)
        return [e[0] for e in arr]

@dataclass
class SetType(Collection):
    elemtype: PqlType


class ColumnType(PqlType):
    def flatten(self):
        return [self]

    is_concrete = True  # Concrete = actually contains data
    primary_key = False
    readonly = False


@dataclass
class TableType(Collection):
    name: str
    columns: Dict[str, ColumnType]
    temporary: bool

    def add_column(self, col: ColumnType):
        assert col.name not in self.columns
        self.columns[col.name] = col

    def flatten(self):
        return [atom for col in self.columns.values() for atom in col.flatten()]

    def params(self):
        return [c for c in self.columns.values() if c.is_concrete and not c.readonly]

    def flat_length(self):
        # Maybe memoize
        return len(self.flatten())

    def import_result(self, arr):
        return list(self._import_result(arr))

    def _import_result(self, arr):
        expected_length = self.flat_length()
        for row in arr:
            assert len(row) == expected_length
            i = iter(row)
            s = ({name: col.type.restructure_result(i) for name, col in self.columns.items()})
            yield s



@dataclass
class StructType(Collection):
    name: str
    members: Dict[str, PqlType]

    def restructure_result(self, i):
        return ({name: col.restructure_result(i) for name, col in self.members.items()})


@dataclass
class DatumColumnType(ColumnType):
    name: str
    type: PqlType
    primary_key: bool = False
    readonly: bool = False

@dataclass
class StructColumnType(ColumnType):
    name: str
    type: StructType
    members: Dict[str, ColumnType]

    def flatten(self):
        return [atom for col in self.members.values() for atom in col.flatten()]

@dataclass
class RelationalColumnType(ColumnType):
    name: str
    type: PqlType
    query: Optional[Any] = None # XXX what now?

def make_column(name, type_, query=None):
    if isinstance(type_, StructType):
        assert not query
        return StructColumnType(name, type_, {
            n: make_column(name+"_"+n, m) for (n,m) in type_.members.items()
        })
    elif query or isinstance(type_, TableType):
        return RelationalColumnType(name, type_, query)
    else:
        return DatumColumnType(name, type_)
    assert False, type_