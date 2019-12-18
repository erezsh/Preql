from typing import List, Any, Optional

from .utils import dataclass
from . import pql_types as types
from .exceptions import Meta


@dataclass
class Ast(types.PqlObject):
    meta: Optional[Meta]

class Expr(Ast): pass

class Statement(Ast): pass

from lark import Token


@dataclass
class Name(Expr):
    "Reference to an object (table, tabledef, column (in `where`), instance, etc.)"
    name: (Token, str)

    def __repr__(self):
        return f'Name({self.name})'

@dataclass
class Attr(Expr):
    "Reference to an attribute (usually a column)"
    expr: Expr
    name: str

@dataclass
class Const(Expr):
    type: types.PqlType
    value: Any

@dataclass
class Ellipsis(Expr):
    pass

@dataclass
class Compare(Expr):
    op: str
    args: List[types.PqlObject]

@dataclass
class Arith(Expr):
    op: str
    args: List[types.PqlObject]

@dataclass
class Contains(Expr):
    op: str
    args: List[types.PqlObject]

@dataclass
class DescOrder(Expr):
    value: Expr # Column

@dataclass
class Like(Expr):
    str: Expr
    pattern: Expr

@dataclass
class NamedField(Expr):
    name: Optional[str]
    value: types.PqlObject #(Expr, types.PqlType)


class TableOperation(Expr): pass

@dataclass
class Selection(TableOperation):
    table: (Expr, types.PqlType)    # XXX find a single base-class
    conds: List[Expr]

@dataclass
class Projection(TableOperation):
    table: types.PqlObject # (Expr, types.PqlType)    # XXX etc.
    fields: List[NamedField]
    groupby: bool = False
    agg_fields: List[NamedField] = ()

    def __created__(self):
        if self.groupby:
            assert self.fields or self.agg_fields
        else:
            assert self.fields and not self.agg_fields

@dataclass
class Order(TableOperation):
    table: Expr
    fields: List[Expr]

@dataclass
class Update(TableOperation):
    table: Expr
    fields: List[NamedField]

@dataclass
class New(Expr):
    type: str
    args: list   # Func args

@dataclass
class FuncCall(Expr):
    func: Any   # objects.Function ?
    args: list   # Func args

@dataclass
class Type(Ast):
    name: str
    nullable: bool = False

class Definition:
    pass

@dataclass
class ColumnDef(Ast, Definition):
    name: str
    type: Type
    query: Optional[Expr] = None


@dataclass
class TableDef(Statement, Definition):
    name: str
    columns: List[ColumnDef] = ()
    # meta: object

@dataclass
class StructDef(Statement, Definition):
    name: str
    members: list

@dataclass
class VarDef(Statement):
    name: str
    value: Expr

@dataclass
class FuncDef(Statement):
    userfunc: types.PqlObject   # XXX Why not use UserFunction?


@dataclass
class Print(Statement):
    value: Expr

@dataclass
class Return(Statement):
    value: Expr

@dataclass
class Throw(Statement):
    value: Expr

@dataclass
class CodeBlock(Statement):
    statements: List[Ast]

@dataclass
class Try(Statement):
    try_: CodeBlock
    catch_expr: Expr
    catch_block: CodeBlock

@dataclass
class If(Statement):
    cond: Expr
    then: CodeBlock
    else_: Optional[CodeBlock] = None


