from enum import Enum
from dataclasses import dataclass as _dataclass

dataclass = _dataclass

class Dataclass:
    def __post_init__(self):
        if not hasattr(self, '__annotations__'):
            return
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            assert value is None or isinstance(value, type_), (name, value, type_)

class Ast(Dataclass):
    pass



@dataclass
class Type(Dataclass):
    pass

class ValueType(Type):
    pass

class AnyType(ValueType):
    pass

class TabularType(ValueType):
    pass

@dataclass
class RelationalType(Type):  # TablularType?
    "Signifies a relationship between tables"

    table_name: str
    column_name: str = 'id'
    # backref_name: str


class AtomType(ValueType):
    pass

class BoolType(AtomType):
    pass

class IntegerType(AtomType):
    pass

@dataclass
class IdType(AtomType): # IntegerType?
    table: str

class StringType(AtomType):
    pass

class NullType(AtomType):
    pass

@dataclass
class ArrayType(ValueType):
    elem_type: ValueType




##################################################
#                Expressions
##################################################

class Expr(Ast):
    type: Type = None

@dataclass
class Arith(Expr):
    op: str
    exprs: list

@dataclass
class Compare(Expr):
    op: str
    exprs: list

@dataclass
class Value(Expr):
    value: object
    type: ValueType

    @classmethod
    def from_pyobj(cls, obj):
        if obj is None:
            return cls(None, NullType())
        if isinstance(obj, str):
            return cls(obj, StringType())
        assert False

@dataclass
class Identifier(Expr):
    "Any reference; Prior to type resolution"
    name: list
    resolved: object = None

    def __repr__(self):
        if self.resolved:
            return "Identifier(%s, resolved=%r)" % ('.'.join(self.name), self.resolved)

        return "Identifier(%s)" % '.'.join(self.name)


class TabularExpr(Expr):
    type = TabularType

@dataclass
class Join(TabularExpr):
    exprs: list

    __types__ = {
        'exprs': [TabularType]
    }

@dataclass
class FreeJoin(Join):
    pass

@dataclass
class AutoJoin(Join):
    pass


@dataclass
class Projection(TabularExpr):
    tab: Expr
    exprs: list

    __types__ = {
        'tab': TabularType,
    }

@dataclass
class Selection(TabularExpr):
    tab: Expr
    exprs: list

    __types__ = {
        'tab': TabularType,
        'exprs': [BoolType]
    }


@dataclass
class FuncArgs(Expr):
    pos_args: list
    named_args: dict

    @property
    def exprs(self):
        return self.pos_args + self.named_args.values()



@dataclass
class FuncCall(Expr):
    name: str
    args: FuncArgs
    resolved: object = None

    @property
    def exprs(self):
        return [self.args]


##################################################
#                 Declarations
##################################################

class Declaration(Ast):
    pass

@dataclass
class Function(Declaration):
    name: str
    params: list
    expr: Expr

    param_types: list = None
    return_type: Type = None


@dataclass
class Table(Declaration):
    name: str
    columns: list

@dataclass
class Column(Declaration):
    name: str
    type: Type
    backref: str  # Handled in type?
    is_nullable: bool
    is_pk: bool

    table: Table = None


### Declaration references, resolved by identifier

class DeclRef(Expr):
    "Objects that refer to declared objects"
    pass

@dataclass
class ColumnRef(Expr):
    # tab: TabularExpr
    column: Column

    @property
    def type(self):
        return self.column.type

# @dataclass    # Query?
# class RowRef(Expr):
#     tab: TabularExpr
#     row_id: int

#     type: TabularType

##################################################
#                 Statements
##################################################

class Stmt(Ast):
    pass

@dataclass
class AddRow(Stmt):
    table: str
    args: list
    as_: str
