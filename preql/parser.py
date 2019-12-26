from lark import Lark, Transformer, v_args, UnexpectedInput, UnexpectedToken

from .exceptions import pql_SyntaxError, Meta
from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects

# class Name(str):
class Name(str):
    def __new__(cls, value, meta):
        obj = str.__new__(cls, value)
        obj.meta = meta
        return obj

def token_value(self, meta, t):
    # return Name(str(t), meta_from_token(t))
    return Name(str(t), meta)

@v_args(inline=False)
def as_list(_, args):
    return args

def meta_from_token(text, tok):
    return Meta(
        text,
        tok.pos_in_stream,
        tok.line,
        tok.column,
        tok.end_pos,
        tok.end_line,
        tok.end_column,
    )

def meta_d(text, meta):
    return Meta(
        text,
        meta.start_pos,
        meta.line,
        meta.column,
        meta.end_pos,
        meta.end_line,
        meta.end_column,
    )

def _args_wrapper(f, data, children, meta):
    return f(meta_d(f.__self__.code, meta), *children)

@v_args(wrapper=_args_wrapper)
class T(Transformer):
    def __init__(self, code):
        self.code = code

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
        return objects.List_(meta_d(self.code, meta), items)

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

    # compare_op = token_value
    # arith_op = token_value
    add_op = token_value
    mul_op = token_value
    comp_op = token_value
    # contains_op = as_list

    def compare(self, meta, a, op, b):
        return ast.Compare(meta, op, [a,b])

    def _arith_expr(self, meta, a, op, b):
        return ast.Arith(meta, op, [a,b])

    add_expr = _arith_expr
    term = _arith_expr
    power = _arith_expr

    # def contains(self, meta, a, op, b):
    #     return ast.Contains(meta, " ".join(op), [a,b])

    like = ast.Like
    var = ast.Name
    getattr = ast.Attr
    named_expr = ast.NamedField
    order = ast.Order
    update = ast.Update
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
    return_stmt = ast.Return
    throw = ast.Throw
    if_stmt = ast.If
    try_catch = ast.Try

    # def ellipsis(self, meta):
    #     return ast.Ellipsis(meta)
    ellipsis = ast.Ellipsis

    @v_args(inline=False, meta=True)
    def codeblock(self, stmts, meta):
        return ast.CodeBlock(meta_d(self.code, meta), stmts)

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
        m = Meta(s,
                 e.pos_in_stream, e.line, e.column,
                 e.pos_in_stream, e.line, e.column)
        # m = m.remake(parent=m)
        # raise pql_SyntaxError(m, str(e) + e.get_context(s)) from e
        if isinstance(e, UnexpectedToken):
            msg = "Unexpected token: '%s'" % e.token
        else:
            msg = "Unexpected character: '%s'" % s[e.pos_in_stream]
        raise pql_SyntaxError(m, "Syntax error: " + msg)

    # print(tree)

    return T(code=s).transform(tree)

def parse_expr(s):
    tree = parser.parse(s, start="expr")
    return T(code=s).transform(tree)
