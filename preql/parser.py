from ast import literal_eval

from lark import Lark, Transformer, v_args, UnexpectedInput, UnexpectedToken, Token

from .exceptions import pql_SyntaxError, Meta
from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects

class Str(str):
    def __new__(cls, value, meta):
        obj = str.__new__(cls, value)
        obj.meta = meta
        return obj

def token_value(self, meta, t):
    return Str(str(t), meta)

@v_args(inline=False)
def as_list(_, args):
    return args

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
    "Create meta with 'code' from transformer"
    return f(meta_d(f.__self__.code, meta), *children)


# Taken from Lark (#TODO provide it in lark utils?)
def _fix_escaping(s):
    w = ''
    i = iter(s)
    for n in i:
        w += n
        if n == '\\':
            try:
                n2 = next(i)
            except StopIteration:
                raise ValueError("Literal ended unexpectedly (bad escaping): `%r`" % s)
            if n2 == '\\':
                w += '\\\\'
            elif n2 not in 'uxnftr':
                w += '\\'
            w += n2
    w = w.replace('\\"', '"').replace("'", "\\'")

    to_eval = "u'''%s'''" % w
    try:
        s = literal_eval(to_eval)
    except SyntaxError as e:
        raise ValueError(s, e)

    return s

@v_args(wrapper=_args_wrapper)
class T(Transformer):
    def __init__(self, code):
        self.code = code

    name = token_value

    def string(self, meta, s):
        return ast.Const(meta, types.String, _fix_escaping( s.value[1:-1]) )

    def int(self, meta, i):
        return ast.Const(meta, types.Int, int(i))

    def float(self, meta, f):
        return ast.Const(meta, types.Float, float(f))

    def null(self, meta):
        return ast.Const(meta, types.null, None)
    def false(self, meta):
        return ast.Const(meta, types.Bool, False)
    def true(self, meta):
        return ast.Const(meta, types.Bool, True)

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

    add_op = token_value
    mul_op = token_value
    comp_op = token_value

    def compare(self, meta, a, op, b):
        return ast.Compare(meta, op, [a,b])

    def _arith_expr(self, meta, a, op, b):
        return ast.Arith(meta, op, [a,b])

    def or_test(self, meta, a, b):
        return ast.Or(meta, [a, b])

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
    delete = ast.Delete
    desc = ast.DescOrder
    new = ast.New
    new_rows = ast.NewRows
    func_call = ast.FuncCall
    range = ast.Range

    selection = ast.Selection
    projection = ast.Projection
    slice = ast.Slice

    def projection_grouped(self, meta, table, keys, values):
        return ast.Projection(meta, table, keys, True, values)

    def projection_grouped_nokeys(self, meta, table, values):
        return ast.Projection(meta, table, [], True, values)

    def projection_grouped_novalues(self, meta, table, keys):
        return ast.Projection(meta, table, keys, True, [])

    # Statements / Declarations
    param = objects.Param
    func_def = lambda self, meta, *args: ast.FuncDef(meta, objects.UserFunction(*args))
    set_value = ast.SetValue
    insert_rows = ast.InsertRows
    struct_def = ast.StructDef
    table_def = ast.TableDef
    col_def = ast.ColumnDef
    member_def = as_list
    print = ast.Print
    return_stmt = ast.Return
    throw = ast.Throw
    if_stmt = ast.If
    try_catch = ast.Try
    one = lambda self, meta, nullable, expr: ast.One(meta, expr, nullable is not None)

    def table_def_by_expr(self, meta, name, table_expr):
        return ast.SetValue(meta, ast.Name(meta, name), ast.FuncCall(meta, ast.Name(meta, 'temptable'), [table_expr]))

    @v_args(inline=False)
    def exclude(self, names):
        return [Str(n.lstrip('!'), n.meta) for n in names]

    exclude_name = token_value

    def ellipsis(self, meta, exclude=None):
        return ast.Ellipsis(meta, exclude or [])

    @v_args(inline=False, meta=True)
    def codeblock(self, stmts, meta):
        return ast.CodeBlock(meta_d(self.code, meta), stmts)

    # @v_args(meta=True)
    # def table_def(self, args, meta):
    #     return ast.TableDef(*args, meta)

    def __default__(self, data, children, meta):
        raise Exception("Unknown rule:", data)


class Postlexer:
    def process(self, stream):
        paren_level = 0
        for token in stream:
            if not (paren_level and token.type == '_NL'):
                yield token

            if token.type == 'LPAR':
                paren_level += 1
            elif token.type == 'RPAR':
                paren_level -= 1
                assert paren_level >= 0

    # XXX Hack for ContextualLexer. Maybe there's a more elegant solution?
    @property
    def always_accept(self):
        return ('_NL',)


parser = Lark.open(
    'preql.lark',
    rel_to=__file__,
    parser='lalr',
    postlex=Postlexer(),
    start=['stmts', 'expr'],
    maybe_placeholders=True,
    propagate_positions=True,
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
