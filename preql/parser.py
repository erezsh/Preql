from lark import Lark, Transformer, v_args, UnexpectedInput

from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects
# from .exceptions import pql_SyntaxError

def token_value(_, _2, t):
    return t #.value

@v_args(inline=False)
def as_list(_, args):
    return args

def meta_d(meta):
    return {
        'line': meta.line,
        'column': meta.column,
    }

def _args_wrapper(f, data, children, meta):
    return f(meta_d(meta), *children)

@v_args(wrapper=_args_wrapper)
class T(Transformer):
    name = token_value

    def string(self, meta, s):
        return ast.Const(meta, types.String, s.value[1:-1])

    def int(self, meta, i):
        return ast.Const(meta, types.Int, int(i))

    def float(self, meta, f):
        return ast.Const(meta, types.Float, float(f))

    def null(self, meta):
        return ast.Const(meta, types.null, None)

    @v_args(inline=False, meta=True)
    def list(self, items, meta):
        return objects.List_(meta_d(meta), items)

    expr_list = as_list
    proj_exprs = as_list
    arguments = as_list
    func_params = as_list
    func_args = as_list
    col_defs = as_list
    member_defs = as_list
    stmts = as_list


    # types
    def typemod(self, meta, *args):
        return [t.value for t in args]
    def type(self, meta, name, mods):
        # TODO pk
        return ast.Type(meta, name, '?' in (mods or ''))

    compare_op = token_value
    arith_op = token_value
    contains_op = as_list

    def compare(self, meta, a, op, b):
        return ast.Compare(meta, op, [a,b])

    def arith_expr(self, meta, a, op, b):
        return ast.Arith(meta, op, [a,b])

    def contains(self, meta, a, op, b):
        return ast.Contains(meta, " ".join(op), [a,b])

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

    def projection_grouped(self, meta, table, keys, values):
        return ast.Projection(meta, table, keys, True, values)

    def projection_grouped_nokeys(self, meta, table, values):
        return ast.Projection(meta, table, [], True, values)

    def projection_grouped_novalues(self, meta, table, keys):
        return ast.Projection(meta, table, keys, True, [])

    # Statements / Declarations
    param = objects.Param
    func_def = lambda self, meta, *args: ast.FuncDef(meta, objects.UserFunction(*args))
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
