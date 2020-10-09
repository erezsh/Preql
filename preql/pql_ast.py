from typing import List, Any, Optional, Dict, Union

from .utils import dataclass, TextReference
from .pql_types import Type, Object
from .types_impl import repr_value

# TODO We want Ast to typecheck, but sometimes types are still unknown (i.e. at parse time).
# * Use incremental type checks?
# * Use two tiers of Ast?


@dataclass
class Ast(Object):
    text_ref: Optional[TextReference]

class Expr(Ast): pass

@dataclass
class Marker(Expr):
    pass

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
    type: Type

@dataclass
class ResolveParameters(Expr):
    obj: Object
    values: Dict[str, Object]

    type = Ast  # XXX Not the place!

@dataclass
class ResolveParametersString(Expr):
    type: Object
    string: Object
    # values: Dict[str, Object]
    # state: Any


@dataclass
class Attr(Expr):
    "Reference to an attribute (usually a column)"
    expr: Optional[Object] #Expr
    name: Union[str, Marker]

@dataclass
class Const(Expr):
    type: Type
    value: Any

    def repr(self, state):
        return repr_value(state, self)

@dataclass
class Ellipsis(Expr):
    exclude: List[Union[str, Marker]]

@dataclass
class Compare(Expr):
    op: str
    args: List[Object]

@dataclass
class Arith(Expr):
    op: str
    args: List[Object]

@dataclass
class Or(Expr):
    args: List[Object]

@dataclass
class And(Expr):
    args: List[Object]

@dataclass
class Not(Expr):
    expr: Expr

@dataclass
class Neg(Expr):
    expr: Expr

@dataclass
class Contains(Expr):
    op: str
    args: List[Object]

@dataclass
class DescOrder(Expr):
    value: Expr # Column

@dataclass
class Like(Expr):
    str: Expr
    pattern: Expr

    op = "~"

@dataclass
class NamedField(Expr):
    name: Optional[str]
    value: Object #(Expr, types.PqlType)

@dataclass
class InlineStruct(Expr):
    struct: Expr


class TableOperation(Expr): pass

@dataclass
class Selection(TableOperation):
    table: Object
    conds: List[Expr]

@dataclass
class Projection(TableOperation):
    table: Object
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
    table: Object
    fields: List[Expr]

@dataclass
class Update(TableOperation):
    table: Object
    fields: List[NamedField]

@dataclass
class Delete(TableOperation):
    table: Object
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
    expr: Object
    nullable: bool = False

@dataclass
class Slice(TableOperation):
    table: Object
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
    userfunc: Object   # XXX Why not use UserFunction?


@dataclass
class TableDef(Statement, Definition):
    name: str
    columns: List[Union[ColumnDef, Ellipsis]]
    methods: list

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
    value: Object

@dataclass
class Assert(Statement):
    cond: Object

@dataclass
class Return(Statement):
    value: Object

@dataclass
class Throw(Statement):
    value: Object

@dataclass
class Import(Statement):
    module_path: str
    as_name: Optional[str] = None
    use_core: bool = True

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
    cond: Object
    then: Statement
    else_: Optional[Statement] = None

@dataclass
class For(Statement):
    var: str
    iterable: Object
    do: CodeBlock




# Collections

@dataclass
class List_(Expr):
    type: Object
    elems: list

@dataclass
class Table_Columns(Expr):
    type: Object
    cols: Dict[str, list]

@dataclass
class Dict_(Expr):
    elems: dict


