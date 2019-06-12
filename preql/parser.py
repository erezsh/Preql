from lark import Lark, Transformer, Discard, v_args, UnexpectedInput
from lark.indenter import Indenter

from . import ast_classes as ast
from .utils import classify
from . import exceptions

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
expr_parser = _parser('expr')

as_args = v_args(inline=False)

@v_args(inline=True)
class ToAST(Transformer):
    name = str
    as_ = str
    stmts = as_args(list)
    ref = ast.Reference
    getattr = ast.GetAttribute

    # Table definition
    def table_def(self, name, *cols):
        cols = [ast.Column('id', None, False, True, type=ast.IdType(name))] + list(cols)
        cols_dict = {c.name: c for c in cols}
        assert len(cols) == len(cols_dict)
        table = ast.TableDef(name, cols_dict)
        for c in cols:
            assert c.table is None
            c.table = table
        return table

    typemod = as_args(list)
    def col_def(self, name, type_, typemod, backref):
        if backref:
            assert isinstance(type_, ast.RelationalType) # TODO nice error
        return ast.Column(name, backref, typemod and '?' in typemod, False, type=type_)

    # Add Row
    arguments = as_args(list)
    def add_row(self, table, args, as_name):
        return ast.AddRow(table, args, as_name)

    def new(self, table, args):
        return ast.NewRow(table, args)

    assign = as_args(tuple)
    assigns = as_args(list)

    def order_asc(self, expr):
        return ast.OrderSpecifier(expr, True)
    def order_desc(self, expr):
        return ast.OrderSpecifier(expr, False)

    proj_exprs = as_args(list)
    sel_exprs = as_args(list)

    projection = ast.Projection
    selection = ast.Selection

    # Query
    # def query(self, table, *elems):

    #     d = classify(elems, lambda e: e.data)
    #     proj_asts = d.pop('projection', [])
    #     sel_asts = d.pop('selection', [])
    #     order_asts = d.pop('order', [])
    #     # func_trees = d.pop('query_user_func', [])
    #     assert not d, d
    #     if len(proj_asts) > 1:
    #         raise Exception("Specified more than one projection for the same table")
    #     if len(order_asts) > 1:
    #         raise Exception("Specified more than one order for the same table")
    #     # if len(func_trees) > 1:
    #     #     raise Exception("Specified more than one limit for the same table")

    #     if proj_asts:
    #         projections, aggregates = proj_asts[0].children
    #     else:
    #         projections = aggregates = None

    #     order = order_asts[0].children if order_asts else []
    #     selections = [cmp for sel in sel_asts for cmp in sel.children]
    #     # funcs = [func.children[0] for func in func_trees]

    #     obj = ast.Query(table, selections, projections or [], order, aggregates or [])
        # for f in funcs:
        #     if f.args.named_args:
        #         raise NotImplementedError("No support for named args yet in this scenario")
        #     new_args = ast.FuncArgs([obj] + f.args.pos_args, {})
        #     obj = ast.FuncCall(f.name, new_args)
        
        # return obj

    def range(self, start, end):
        if start:
            start = ast.Value(start, ast.IntegerType())
        if end:
            end = ast.Value(end, ast.IntegerType())
        return ast.Range(start, end)

    def arith_op(self, op):
        return op

    # selection = as_args(list)
    # func_args = as_args(list)
    # projection = as_args(list)
    func_params = as_args(list)
    func_def = ast.FunctionDef
    func_call = ast.FuncCall

    named_expr = ast.NamedExpr #as_args(tuple)
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
        return ast.FuncArgs(pos_args, named_args)
        
    # Atoms (Types and Values)
    def string(self, x):
        return ast.Value.from_pyobj(x[1:-1])
    def null(self):
        return ast.Value.from_pyobj(None)
    def int(self, num):
        return ast.Value.from_pyobj(int(num))
    def float(self, num):
        return ast.Value.from_pyobj(float(num))


    @as_args
    def array(self, v):
        return ast.Value(v, ast.ArrayType(ast.AnyType()))

    typename = str
    def type(self, typename):
        # return Type.from_str(typename), typemod
        try:
            return {
                "integer": ast.IntegerType,
                "string": ast.StringType,
                "float": ast.FloatType,
            }[typename]()
        except KeyError:
            return ast.RelationalType(typename)

    # Operations
    compare_op = str
    def compare(self, a, op, b):
        return ast.Compare(op, [a, b])

    def arith_expr(self, a, op, b):
        return ast.Arith(op, [a, b])

    neg = ast.Neg
    desc = ast.Desc

def parse(s):
    t = parser.parse(s.rstrip() + '\n')
    t = ToAST().transform(t)
    return t

def parse_expr(q):
    try:
        t = expr_parser.parse(q.strip())
    except UnexpectedInput as e:
        message = e.match_examples(expr_parser.parse, {
            'Unexpected expression. Did you forget an operator?': ['a 1', '1 1', 'a a', 'a[b] 1'],
            'Misplaced operator': ['[]', '{}', '()', 'a[{}]', 'a[()]', 'a({})', 'a([])', 'a{()}', 'a{[]}'],
            'Mismatched bracket': ['a[)', 'a{)', 'a(]'],
            'Unclosed bracket': ['a[', 'a{', 'a(', 'a[[', 'a{{', 'a(('],
            'Superfluous closing bracket': [']', 'a]'],
            'Selection is empty (a condition is required)': ['a[]'],
            'Projection is empty (an expression is required)': ['a{}'],
        })
        if not message:
            raise
        raise exceptions.PreqlError_Syntax('Syntax Error: ' + message, e.get_context(q), e.line, e.column)


    t = ToAST().transform(t)
    return t


def test():
    # a = open("preql/simple1.pql").read()
    # a = open("preql/simple2.pql").read()
    a = open("preql/tree.pql").read()
    for s in parse(a):
        print(s)

if __name__ == '__main__':
    test()