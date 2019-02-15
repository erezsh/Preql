from enum import Enum
from dataclasses import dataclass as _dataclass

dataclass = _dataclass

from lark import Tree

class Dataclass:
    def __post_init__(self):
        if not hasattr(self, '__annotations__'):
            return
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            assert value is None or isinstance(value, type_), (name, value, type_)

class Ast(Dataclass):
    def _to_tree(self, expr):
        if isinstance(expr, Ast):
            return expr.to_tree()
        return expr

    def to_tree(self):
        if getattr(self, 'resolved', False):
            return self.resolved.to_tree()

        children = []
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            if isinstance(value, list):
                t = Tree('.'+name, [self._to_tree(v) for v in value])
            else:
                t = Tree('.'+name, [self._to_tree(value)])

            children.append(t)
        return Tree(self.__class__.__name__, children)
    pass



@dataclass(frozen=True)
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

@dataclass
class BackRefType(Type):
    to_table: str


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

    type = BoolType()

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

class Resolvable:
    @property
    def type(self):
        if self.resolved:
            return self.resolved.type

    @property
    def resolved_table(self):
        rt = self.resolved.resolved_table
        assert rt, self.resolved
        return rt

@dataclass
class Identifier(Expr, Resolvable):
    "Any reference; Prior to type resolution"
    name: list
    resolved: object = None

    def __repr__(self):
        if self.resolved:
            return "Identifier(%s, resolved=%r)" % ('.'.join(self.name), self.resolved)

        return "Identifier(%s)" % '.'.join(self.name)


class TabularExpr(Expr):
    type = TabularType()
    resolved_table = None

@dataclass
class Table(TabularExpr):
    columns: list

    def __getitem__(self, name):
        cols = [c for c in self.columns if c.name == name]
        if not cols:
            raise KeyError("No such column: %s" % name)
        col ,= cols
        return col

    @property
    def relations(self):
        return [c for c in self.columns if isinstance(c.type, RelationalType)]

    @property
    def id(self):
        x ,= [c for c in self.columns if isinstance(c.type, IdType)]
        return x

    @property
    def resolved_table(self):
        return self


@dataclass
class Join(TabularExpr):
    exprs: list

    __types__ = {
        'exprs': [TabularType]
    }

    def __getitem__(self, name):
        exprs = [c for c in self.exprs if c.name == name]
        if not exprs:
            raise KeyError("No such alias: %s" % name)
        expr ,= exprs
        return expr

@dataclass
class FreeJoin(Join):
    pass

@dataclass
class AutoJoin(Join):
    pass


@dataclass
class Projection(TabularExpr):
    table: Expr
    exprs: list

    __types__ = {
        'table': TabularType,
    }

@dataclass
class Selection(TabularExpr):
    table: Expr
    exprs: list

    __types__ = {
        'table': TabularType,
        'exprs': [BoolType]
    }

    @property
    def type(self):
        return self.table.type

@dataclass
class AliasedTable(TabularExpr):
    table: Expr
    name: str


@dataclass
class FuncArgs(Expr):
    pos_args: list
    named_args: dict

    @property
    def exprs(self):
        return self.pos_args + self.named_args.values()

@dataclass
class NamedExpr(Expr):
    name: str   # nullable
    expr: Expr


@dataclass
class FuncCall(Expr, Resolvable):
    name: str
    args: FuncArgs
    resolved: object = None

    @property
    def exprs(self):
        return [self.args]


@dataclass
class Count(Expr):
    exprs: list
    type: IntegerType

##################################################
#                 Declarations
##################################################

class Stmt(Ast):
    pass

@dataclass
class AddRow(Stmt):
    table: str
    args: list
    as_: str

class Declaration(Stmt):
    pass

@dataclass
class Function(Declaration):
    name: str
    params: list
    expr: Expr

    param_types: list = None
    return_type: Type = None


@dataclass
class NamedTable(Table, Declaration):
    name: str

    def __repr__(self):
        return '<NamedTable:%s>' % self.name

@dataclass
class Column(Expr, Declaration):
    name: str
    backref: str  # Handled in type?
    is_nullable: bool
    is_pk: bool

    type: Type
    table: Table = None

    def to_tree(self):
        return '-> %s.%s' % (self.table.name, self.name)



@dataclass
class RowRef(Expr):
    table: Expr
    row_id: int


### Declaration references, resolved by identifier

# class DeclRef(Expr):
#     "Objects that refer to declared objects"
#     pass

# @dataclass
# class ColumnRef(Expr):
#     # tab: TabularExpr
#     column: Column

#     @property
#     def type(self):
#         return self.column.type

#     def __repr__(self):
#         return 'ColumnRef:%s' % self.column.name


# @dataclass
# class TableRef(TabularExpr):
#     table: Table

#     type = TabularType

#     def __repr__(self):
#         return 'TableRef:%s' % self.table.name

#     def __getitem__(self, name):
#         col ,= [c for c in self.table if c.name == name]
#         return col


# @dataclass    # Query?
# class RowRef(Expr):
#     tab: TabularExpr
#     row_id: int

#     type: TabularType

##################################################
#                 Statements
##################################################
