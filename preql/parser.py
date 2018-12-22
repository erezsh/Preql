from lark import Lark, Transformer, Discard, v_args
from lark.indenter import Indenter

from .utils import classify
from .ast_classes import *

class PythonIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = ['LPAR', 'LSQB', 'LBRACE']
    CLOSE_PAREN_types = ['RPAR', 'RSQB', 'RBRACE']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8

parser = Lark.open('preql/preql.lark', parser='lalr', postlex=PythonIndenter(), lexer='standard', maybe_placeholders=True)
query_parser = Lark.open('preql/preql.lark', start='query', parser='lalr', postlex=PythonIndenter(), lexer='standard', maybe_placeholders=True)


as_args = v_args(inline=False)

@v_args(inline=True)
class ToAST(Transformer):
    name = str
    as_ = str
    stmts = as_args(list)

    # Table definition
    def table_def(self, name, *cols):
        cols = [Column('id', Type.from_str('Int'), None, False, True)] + list(cols)
        return Table(name, cols)

    typemod = as_args(list)
    def col_def(self, name, type_, backref):
        type_, typemod = type_
        if backref:
            assert isinstance(type_, TableType) # TODO nice error
        return Column(name, type_, backref, typemod and '?' in typemod, False)

    # Add Row
    arguments = as_args(list)
    def add_row(self, table, args, as_name):
        return AddRow(TableType(table), args, as_name)

    assign = as_args(tuple)
    assigns = as_args(list)

    # Query
    query = Query
    selection = as_args(list)
    func_args = as_args(list)
    projection = as_args(list)
    func_params = as_args(list)
    func_def = Function
    func_call = FuncCall

    # Atoms (Types and Values)
    def string(self, x):
        return Value(StrType, x[1:-1])

    typename = str
    def type(self, typename, typemod):
        return Type.from_str(typename), typemod

    ref = as_args(Ref)

    # Operations
    compare_op = str
    def compare(self, a, op, b):
        return Compare(str(op), [a, b])

    def arith_expr(self, a, op, b):
        return Arith(str(op), [a, b])

def parse(s):
    t = parser.parse(s.rstrip() + '\n')
    t = ToAST().transform(t)
    return t

def parse_query(q):
    t = query_parser.parse(q.strip())
    # t = ToAST().transform(t)
    return t


def test():
    a = open("preql/simple1.pql").read()
    # a = open("preql/simple2.pql").read()
    # a = open("preql/tree.pql").read()
    for s in parse(a):
        print(s)

test()