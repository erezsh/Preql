from lark import Lark, Transformer, Discard, v_args
from lark.indenter import Indenter

from .ast_classes import *

class PythonIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = ['LPAR', 'LSQB', 'LBRACE']
    CLOSE_PAREN_types = ['RPAR', 'RSQB', 'RBRACE']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8


def _parser(start):
    return Lark.open('preql.lark', rel_to=__file__, start=start, parser='lalr', postlex=PythonIndenter(), lexer='standard', maybe_placeholders=True)

parser = _parser('start')
query_parser = _parser('query')

as_args = v_args(inline=False)

@v_args(inline=True)
class ToAST(Transformer):
    name = str
    as_ = str
    stmts = as_args(list)

    # Table definition
    def table_def(self, name, *cols):
        cols = [Column('id', None, False, True, type=IdType(name))] + list(cols)
        table = NamedTable(cols, name)
        for c in cols:
            assert c.table is None
            c.table = table
        return table

    typemod = as_args(list)
    def col_def(self, name, type_, typemod, backref):
        if backref:
            assert isinstance(type_, RelationalType) # TODO nice error
        return Column(name, backref, typemod and '?' in typemod, False, type=type_)

    # Add Row
    arguments = as_args(list)
    def add_row(self, table, args, as_name):
        return AddRow(table, args, as_name)

    assign = as_args(tuple)
    assigns = as_args(list)

    # Query
    def query(self, tab, sel, proj):
        if sel:
            tab = Selection(tab, sel)
        if proj:
            tab = Projection(tab, proj)
        return tab

    def query2(self, tab, proj, sel):
        return self.query(tab, sel, proj)

    selection = as_args(list)
    # func_args = as_args(list)
    projection = as_args(list)
    func_params = as_args(list)
    func_def = Function
    func_call = FuncCall

    named_expr = NamedExpr #as_args(tuple)
    # def func_arg(self, name, value):
    #     return name, value

    @as_args
    def func_args(self, args):
        pos_args = []
        named_args = {}
        for ne in args:
            if ne.name:  # Named arg
                assert ne.name not in named_args
                named_args[ne.name] = ne.expr
            else:
                assert not named_args
                pos_args.append(ne.expr)
        return FuncArgs(pos_args, named_args)
        
    # Atoms (Types and Values)
    def string(self, x):
        return Value.from_pyobj(x[1:-1])

    def null(self):
        return Value.from_pyobj(None)

    @as_args
    def array(self, v):
        return Value(v, ArrayType(AnyType()))

    typename = str
    def type(self, typename):
        # return Type.from_str(typename), typemod
        try:
            return {
                "Int": IntegerType,
                "Str": StringType,
            }[typename]()
        except KeyError:
            return RelationalType(typename)

    identifier = as_args(Identifier)

    # Operations
    compare_op = str
    def compare(self, a, op, b):
        return Compare(op, [a, b])

    def arith_expr(self, a, op, b):
        return Arith(op, [a, b])

def parse(s):
    t = parser.parse(s.rstrip() + '\n')
    t = ToAST().transform(t)
    return t

def parse_query(q):
    t = query_parser.parse(q.strip())
    # t = ToAST().transform(t)
    return t


def test():
    # a = open("preql/simple1.pql").read()
    # a = open("preql/simple2.pql").read()
    a = open("preql/tree.pql").read()
    for s in parse(a):
        print(s)

if __name__ == '__main__':
    test()