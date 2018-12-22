from enum import Enum
from dataclasses import dataclass as _dataclass

# from validateit import typeforced
# dataclass = lambda c: typeforced(_dataclass(c))

dataclass = _dataclass

class Ast:
    def __post_init__(self):
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            assert value is None or isinstance(value, type_), (name, value, type_)


BuiltinTypes = Enum('BuiltinTypes', 'Str Int Float')
TypeEnum = Enum('Types', 'builtin table')

class Type(Ast):

    @classmethod
    def from_str(cls, s):
        try:
            return {
                "Int": IntType,
                "Str": StrType,
            }[s]
        except KeyError:
            return TableType(s)

    def __repr__(self):
        return type(self).__name__

@dataclass(frozen=True)
class BuiltinType(Type):
    name: str

    def __repr__(self):
        return self.name

StrType = BuiltinType('Str')
IntType = BuiltinType('Int')

class RelationType(Type):
    pass

@dataclass
class TableType(RelationType):
    name: str

    def __repr__(self):
        return 'TableType(%r)' % self.name

@dataclass
class Table(Ast):
    name: str
    columns: list

@dataclass
class Column(Ast):
    name: str
    type: Type
    backref: str
    is_nullable: bool
    is_pk: bool

@dataclass
class AddRow(Ast):
    table: TableType
    args: list
    as_: str

class Expr(Ast):
    pass

@dataclass
class Arith(Expr):
    op: str
    elems: list

@dataclass
class Value(Expr):
    type: Type
    value: object

@dataclass
class Ref(Expr):
    "Any reference; Prior to type resolution"
    name: list

    def __repr__(self):
        return "Ref(%s)" % '.'.join(self.name)

@dataclass
class Compare(Expr):
    op: str
    elems: list

    def exprs(self):
        return self.elems

@dataclass
class Query(Expr):
    relation: Expr  # ref (table / table.other_table / function / (expr) )
    as_: str
    selection: list
    groupby: list
    projection: list

    def relations(self):
        return [self.relation]

    def exprs(self):
        return self.relations + self.selection + self.projection + self.groupby

@dataclass
class Function(Ast):
    name: str
    params: list
    expr: Expr

@dataclass
class FuncCall(Expr):
    # TODO Are Query and FuncCall the same construct?
    name: str
    args: list

    def exprs(self):
        return self.args