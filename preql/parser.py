from lark import Lark, Transformer, v_args, UnexpectedInput

from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects
# from .exceptions import pql_SyntaxError

def token_value(_, t):
    return t #.value

@v_args(inline=False)
def as_list(_, args):
    return args

@v_args(inline=True)
class T(Transformer):
    name = token_value

    def string(self, s):
        return ast.Const(types.String, s.value[1:-1])

    def int(self, i):
        return ast.Const(types.Int, int(i))

    def float(self, f):
        return ast.Const(types.Float, float(f))

    def null(self):
        return ast.Const(types.null, None)

    list = v_args(inline=False)(objects.List_)

    expr_list = as_list
    proj_exprs = as_list
    arguments = as_list
    func_params = as_list
    func_args = as_list
    col_defs = as_list
    member_defs = as_list
    stmts = as_list


    # types
    def typemod(self, *args):
        return [t.value for t in args]
    def type(self, name, mods):
        # TODO pk
        return ast.Type(name, '?' in (mods or ''))

    compare_op = token_value
    arith_op = token_value
    contains_op = as_list

    def compare(self, a, op, b):
        return ast.Compare(op, [a,b])

    def arith_expr(self, a, op, b):
        return ast.Arith(op, [a,b])

    def contains(self, a, op, b):
        return ast.Contains(" ".join(op), [a,b])

    like = ast.Like
    var = ast.Name
    getattr = ast.Attr
    named_expr = ast.NamedField
    order = ast.Order
    desc = ast.DescOrder
    new = ast.New
    func_call = ast.FuncCall

    selection = ast.Selection
    projection = ast.Projection

    def projection_grouped(self, table, keys, values):
        return ast.Projection(table, keys, True, values)

    def projection_grouped_nokeys(self, table, values):
        return ast.Projection(table, [], True, values)

    def projection_grouped_novalues(self, table, keys):
        return ast.Projection(table, keys, True, [])

    # Statements / Declarations
    param = objects.Param
    func_def = lambda self, *args: ast.FuncDef(objects.UserFunction(*args))
    var_decl = ast.VarDef
    struct_def = ast.StructDef
    table_def = ast.TableDef
    col_def = ast.ColumnDef
    member_def = as_list
    print = ast.Print

    # @v_args(meta=True)
    # def table_def(self, args, meta):
    #     return ast.TableDef(*args, meta)

    def __default__(self, data, children, meta):
        raise Exception("Unknown rule:", data)


parser = Lark.open(
    'preql.lark',
    rel_to=__file__,
    parser='lalr',
    start=['stmts', 'expr'],
    maybe_placeholders=True,
    # transformer=T()
)

def parse_stmts(s):
    try:
        tree = parser.parse(s+"\n", start="stmts")
    except UnexpectedInput as e:
        # raise pql_SyntaxError(e) from e
        raise

    # print(tree)

    return T().transform(tree)

def parse_expr(s):
    tree = parser.parse(s, start="expr")
    return T().transform(tree)
