from typing import List, Any, Optional, Dict, Union
from dataclasses import field

from .utils import dataclass, TextReference, field_list
from .pql_types import Type, Object
from .types_impl import repr_value

# TODO We want Ast to typecheck, but sometimes types are still unknown (i.e. at parse time).
# * Use incremental type checks?
# * Use two tiers of Ast?

@dataclass
class Ast(Object):
    text_ref: Optional[TextReference] = field(init=False, default=None)

    def set_text_ref(self, text_ref):
        object.__setattr__(self, 'text_ref', text_ref)
        return self

class Expr(Ast):
    _args = ()

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
class ParameterizedSqlCode(Expr):
    type: Object
    string: Object
    # values: Dict[str, Object]
    # state: Any


@dataclass
class Attr(Expr):
    "Reference to an attribute (usually a column)"
    expr: Optional[Object] #Expr
    name: Union[str, Marker]

    _args = 'expr',

@dataclass
class Const(Expr):
    type: Type
    value: Any

    def repr(self, state):
        return repr_value(state, self)

@dataclass
class Ellipsis(Expr):
    exclude: List[Union[str, Marker]]

class BinOpExpr(Expr):
    _args = 'args',
class UnaryOpExpr(Expr):
    _args = 'expr',


@dataclass
class Compare(BinOpExpr):
    op: str
    args: List[Object]


@dataclass
class Arith(BinOpExpr):
    op: str
    args: List[Object]

@dataclass
class Or(BinOpExpr):
    args: List[Object]

@dataclass
class And(BinOpExpr):
    args: List[Object]

@dataclass
class Not(UnaryOpExpr):
    expr: Object

@dataclass
class Neg(UnaryOpExpr):
    expr: Object

@dataclass
class Contains(BinOpExpr):
    op: str
    args: List[Object]

@dataclass
class DescOrder(Expr):
    value: Object

    _args = 'value',

@dataclass
class Like(Expr):
    str: Object
    pattern: Object

    op = "~"

    _args = 'str', 'pattern'

@dataclass
class Range(Expr):
    start: Optional[Object]
    stop: Optional[Object]

    _args = 'start', 'stop'

@dataclass
class NamedField(Expr):
    name: Optional[str]
    value: Object #(Expr, types.PqlType)

    _args = 'value',

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
    agg_fields: List[NamedField] = field_list()

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
class Slice(Expr):
    obj: Object
    range: Range

    _args = 'obj',

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
class One(Expr):
    expr: Object
    nullable: bool = False


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
class TableDefFromExpr(Statement, Definition):
    name: str
    expr: Expr
    const: bool


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

@dataclass
class While(Statement):
    cond: Object
    do: Statement



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


