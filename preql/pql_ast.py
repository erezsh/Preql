from typing import List, Any, Optional

from .utils import dataclass
from . import pql_types as types

class Ast: pass

class Expr(Ast): pass

class Statement(Ast): pass


@dataclass
class Name(Expr):
    "Reference to an object (table, tabledef, column (in `where`), instance, etc.)"
    name: str


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
class Compare(Expr):
    op: str
    args: List[Expr]

@dataclass
class Arith(Expr):
    op: str
    args: List[Expr]

@dataclass
class Contains(Expr):
    op: str
    args: List[Expr]

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
    value: Expr


class TableOperation(Expr): pass

@dataclass
class Selection(TableOperation):
    table: (Expr, types.PqlType)    # XXX find a single base-class
    conds: List[Expr]

@dataclass
class Projection(TableOperation):
    table: Expr
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
class New(Expr):
    type: str
    args: list   # Func args

@dataclass
class FuncCall(Expr):
    func: Expr
    args: list   # Func args

@dataclass
class Type:
    name: str
    nullable: bool = False

@dataclass
class ColumnDef:
    name: str
    type: Type
    query: Optional[Expr] = None

class Definition:
    pass

@dataclass
class TableDef(Statement, Definition):
    name: str
    columns: List[ColumnDef]

@dataclass
class StructDef(Statement, Definition):
    name: str
    members: list

@dataclass
class VarDef(Statement):
    name: str
    value: Expr

@dataclass
class Print(Statement):
    value: Expr
