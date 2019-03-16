from copy import copy
from enum import Enum

from lark import Tree

from .utils import Dataclass, dataclass

class Ast(Dataclass):
    def _to_tree(self, expr):
        if isinstance(expr, Ast):
            return expr.to_tree()
        return expr

    def to_tree(self):
        children = []
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            if isinstance(value, list):
                t = Tree('.'+name, [self._to_tree(v) for v in value])
            else:
                t = Tree('.'+name, [self._to_tree(value)])

            children.append(t)
        return Tree(self.__class__.__name__, children)



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

class FloatType(AtomType):
    pass

class RangeType(AtomType):
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
    pass

@dataclass
class Arith(Expr):
    op: str
    exprs: list

@dataclass
class Neg(Expr):
    expr: Expr

@dataclass
class Desc(Expr):
    expr: Expr

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
        elif isinstance(obj, int):
            return cls(obj, IntegerType())
        elif isinstance(obj, float):
            return cls(obj, FloatType())
        assert False

@dataclass
class Range(Expr):
    start: Expr
    end: Expr

    type = RangeType()


@dataclass
class Reference(Expr):
    name: str

@dataclass
class GetAttribute(Expr):
    obj: Expr
    attr: str

@dataclass
class OrderSpecifier(Expr):
    expr: Expr
    asc: bool
    
@dataclass
class Projection(Expr):
    table: Expr
    fields: list
    agg_fields: list

@dataclass
class Selection(Expr):
    table: Expr
    conds: list

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

    @property
    def type(self):
        return self.expr.type


@dataclass
class FuncCall(Expr):
    obj: Expr
    args: FuncArgs



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

##################################################
#                 Declarations
##################################################

class Declaration(Stmt):
    pass

@dataclass
class FunctionDef(Declaration):
    name: str
    params: list
    expr: Expr

    param_types: list = None
    return_type: Type = None


@dataclass
class TableDef(Declaration):
    name: str
    columns: dict

    def __repr__(self):
        return '<TableDef:%s>' % self.name

    # def get_column(self, name):
    #     cols = [c for c in self.columns if c.name == name]
    #     if not cols:
    #         raise AttributeError("No such column: %s" % name)
    #     col ,= cols
    #     return col

    # def cols_by_type(self, type_):
    #     return {name: c for name, c in self.columns.items()
    #             if isinstance(c.type, type_)}

    # def make_alias(self, new_name):
    #     t = AliasedTableDef(new_name, {
    #         name: copy(c) for name, c in self.columns.items()
    #     })
    #     for c in t.columns.values():
    #         c.table = t
    #     return t
    
class AliasedTableDef(TableDef):
    pass

@dataclass
class Column(Expr, Declaration):
    name: str
    backref: str  # Handled in type?
    is_nullable: bool
    is_pk: bool
    type: Type
    table: TableDef = None

    def to_tree(self):
        return '-> %s.%s' % (self.table.name, self.name)



