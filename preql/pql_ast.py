from typing import List, Any, Optional, Dict

from .utils import dataclass
from . import pql_types as types
from .exceptions import Meta

PqlObject = types.PqlObject


@dataclass
class Ast(PqlObject):
    meta: Optional[Meta]

class Expr(Ast): pass

class Statement(Ast): pass


@dataclass
class Name(Expr):
    "Reference to an object (table, tabledef, column (in `where`), instance, etc.)"
    name: str

    def __repr__(self):
        return f'Name({self.name})'

@dataclass
class Parameter(Expr):
    "A typed object without a value"
    name: str
    type: types.PqlType

@dataclass
class ResolveParameters(Expr):
    obj: PqlObject
    values: Dict[str, PqlObject]

    type = Ast  # XXX Not the place!

@dataclass
class ResolveParametersString(Expr):
    type: PqlObject
    string: PqlObject
    # values: Dict[str, PqlObject]
    # state: Any


@dataclass
class Attr(Expr):
    "Reference to an attribute (usually a column)"
    expr: Optional[PqlObject] #Expr
    name: str

@dataclass
class Const(Expr):
    type: types.PqlType
    value: Any

@dataclass
class Ellipsis(Expr):
    exclude: List[str]

@dataclass
class Compare(Expr):
    op: str
    args: List[PqlObject]

@dataclass
class Arith(Expr):
    op: str
    args: List[PqlObject]

@dataclass
class Or(Expr):
    args: List[PqlObject]

@dataclass
class Contains(Expr):
    op: str
    args: List[PqlObject]

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
    value: PqlObject #(Expr, types.PqlType)

@dataclass
class InlineStruct(Expr):
    struct: Expr


class TableOperation(Expr): pass

@dataclass
class Selection(TableOperation):
    table: PqlObject
    conds: List[Expr]

@dataclass
class Projection(TableOperation):
    table: PqlObject
    fields: List[NamedField]
    groupby: bool = False
    agg_fields: List[NamedField] = ()

    def __post_init__(self):
        if self.groupby:
            assert self.fields or self.agg_fields
        else:
            assert self.fields and not self.agg_fields

@dataclass
class Order(TableOperation):
    table: PqlObject
    fields: List[Expr]

@dataclass
class Update(TableOperation):
    table: PqlObject
    fields: List[NamedField]

@dataclass
class Delete(TableOperation):
    table: PqlObject
    conds: List[Expr]

@dataclass
class New(Expr):
    type: str
    args: list   # Func args

@dataclass
class NewRows(Expr):
    type: str
    args: list   # Func args

@dataclass
class FuncCall(Expr):
    func: Any   # objects.Function ?
    args: list   # Func args

@dataclass
class Range(Expr):
    start: Optional[Expr]
    stop: Optional[Expr]

@dataclass
class One(Expr):
    expr: PqlObject
    nullable: bool

@dataclass
class Slice(TableOperation):
    table: Expr
    range: Range

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
    default: Optional[Expr] = None

@dataclass
class FuncDef(Statement, Definition):
    userfunc: types.PqlObject   # XXX Why not use UserFunction?


@dataclass
class TableDef(Statement, Definition):
    name: str
    columns: List[ColumnDef]
    methods: list
    # meta: object

@dataclass
class StructDef(Statement, Definition):
    name: str
    members: list

@dataclass
class SetValue(Statement):
    name: (Name, Attr)
    value: Expr

@dataclass
class InsertRows(Statement):
    name: (Name, Attr)
    value: Expr

@dataclass
class Print(Statement):
    value: PqlObject

@dataclass
class Return(Statement):
    value: PqlObject

@dataclass
class Throw(Statement):
    value: PqlObject

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
    cond: PqlObject
    then: Statement
    else_: Optional[CodeBlock] = None

@dataclass
class For(Statement):
    var: str
    iterable: PqlObject
    do: CodeBlock




# Collections

@dataclass
class List_(Expr):
    type: types.ListType
    elems: list

@dataclass
class Dict_(Expr):
    elems: dict
