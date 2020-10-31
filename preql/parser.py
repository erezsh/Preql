from ast import literal_eval
from pathlib import Path

from lark import Lark, Transformer, v_args, UnexpectedInput, UnexpectedToken, Token

from .utils import TextPos, TextRange, TextReference
from .exceptions import pql_SyntaxError
from . import pql_ast as ast
from . import pql_objects as objects

from .pql_types import T

class Str(str):
    def __new__(cls, value, text_ref):
        obj = str.__new__(cls, value)
        obj.text_ref = text_ref
        return obj

def token_value(self, text_ref, t):
    return Str(str(t), text_ref)


def make_text_reference(text, source_file, meta, children=()):
    ref = TextRange(
        TextPos(
            meta.start_pos,
            meta.line,
            meta.column,
        ),
        TextPos(
            meta.end_pos or meta.start_pos,
            meta.end_line or meta.line,
            meta.end_column or meta.column,
        )
    )

    for c in children:
        if hasattr(c, 'text_ref'):
            assert c.text_ref.context is None
            assert c.text_ref.text is text
            c.text_ref.context = ref

    assert isinstance(source_file, (str, Path)), source_file
    return TextReference(text, str(source_file), ref)



def _args_wrapper(f, data, children, meta):
    "Create meta with 'code' from transformer"
    ref = make_text_reference(*f.__self__.code_ref, meta, children)
    return f(ref, *children)


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

from .compiler import guess_field_name
@v_args(wrapper=_args_wrapper)
class TreeToAst(Transformer):
    def __init__(self, code_ref):
        self.code_ref = code_ref

    name = token_value

    def string(self, meta, s):
        return ast.Const(meta, T.string, _fix_escaping( s.value[1:-1]) )
    def long_string(self, meta, s):
        return ast.Const(meta, T.string, _fix_escaping( s.value[3:-3]) )

    def pql_dict(self, meta, items):
        # if not all(item.name for item in items):
        #     # TODO autocomplete names in some situations, like in projection
        #     raise pql_SyntaxError([meta], "Dict expects all items to have keys")
        d = {}
        for item in items:
            name = item.name or guess_field_name(item.value)
            if name in d:
                raise pql_SyntaxError(meta, f"Dict key appearing more than once: {name}")
            d[name] = item.value

        return ast.Dict_(meta, d)

    def int(self, meta, i):
        return ast.Const(meta, T.int, int(i))

    def float(self, meta, f):
        return ast.Const(meta, T.float, float(f))

    def null(self, meta):
        return ast.Const(meta, T.nulltype, None)
    def false(self, meta):
        return ast.Const(meta, T.bool, False)
    def true(self, meta):
        return ast.Const(meta, T.bool, True)

    @v_args(inline=False, meta=True)
    def pql_list(self, items, meta):
        return ast.List_(make_text_reference(*self.code_ref, meta), T.list[T.any], items)

    @v_args(inline=False)
    def as_list(self, args):
        return args

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

    def and_test(self, meta, a, b):
        return ast.And(meta, [a, b])

    not_test = ast.Not

    add_expr = _arith_expr
    term = _arith_expr
    power = _arith_expr

    neg = ast.Neg
    like = ast.Like
    var = ast.Name
    getattr = ast.Attr
    named_expr = ast.NamedField
    inline_struct = ast.InlineStruct
    order = ast.Order
    update = ast.Update
    delete = ast.Delete
    desc = ast.DescOrder
    new = ast.New
    new_rows = ast.NewRows
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
    param_variadic = objects.ParamVariadic

    def func_def(self, meta, name, params, expr):
        collector = None
        for i, p in enumerate(params):
            if isinstance(p, objects.ParamVariadic):
                if i != len(params)-1:
                    raise pql_SyntaxError(meta, f"A variadic parameter must appear at the end of the function ({p.name})")

                collector = p
                params = params[:-1]

        return ast.FuncDef(meta, objects.UserFunction(name, params, expr, collector))

    def func_call(self, meta, func, args):
        for i, a in enumerate(args):
            if isinstance(a, ast.InlineStruct):
                if i != len(args)-1:
                    raise pql_SyntaxError(meta, f"An inlined struct must appear at the end of the function call ({a})")


        return ast.FuncCall(meta, func, args)

    def set_value(self, meta, lval, rval):
        if not isinstance(lval, (ast.Name, ast.Attr)):
            raise pql_SyntaxError(lval.text_ref, f"{lval.type} is not a valid l-value")
        return ast.SetValue(meta, lval, rval)

    insert_rows = ast.InsertRows
    struct_def = ast.StructDef
    table_def = ast.TableDef
    col_def = ast.ColumnDef
    print = ast.Print
    assert_ = ast.Assert
    return_stmt = ast.Return
    import_stmt = ast.Import
    throw = ast.Throw
    if_stmt = ast.If
    while_stmt = ast.While
    for_stmt = ast.For
    try_catch = ast.Try
    one = lambda self, meta, nullable, expr: ast.One(meta, expr, nullable is not None)

    def marker(self, meta, _marker):
        return ast.Marker(meta)

    def table_def_from_expr(self, meta, const, name, table_expr):
        # c = objects.from_python(bool(const == 'const'))
        # return ast.SetValue(meta, ast.Name(meta, name), ast.FuncCall(meta, ast.Name(meta, 'temptable'), [table_expr, c]))
        return ast.TableDefFromExpr(meta, name, table_expr, const == 'const')

    def ellipsis(self, meta, *exclude):
        return ast.Ellipsis(meta, list(exclude))

    @v_args(inline=False, meta=True)
    def codeblock(self, stmts, meta):
        return ast.CodeBlock(make_text_reference(*self.code_ref, meta), stmts)


    def __default__(self, data, children, meta):
        raise Exception("Unknown rule:", data)


class Postlexer:
    def process(self, stream):
        paren_level = []
        for token in stream:
            if not (paren_level and paren_level[-1] == 'LPAR' and token.type == '_NL'):
                assert token.end_pos is not None
                yield token

            if token.type in ('LPAR', 'LSQB', 'LBRACE'):
                paren_level.append(token.type)
            elif token.type in ('RPAR', 'RSQB', 'RBRACE'):
                p = paren_level.pop()
                assert p == 'L' + token.type[1:]

    # XXX Hack for ContextualLexer. Maybe there's a more elegant solution?
    @property
    def always_accept(self):
        return ('_NL',)


# Prevent expressions like (1and1) or (1ina)
# Changing these terminals in the grammar will prevent collision detection
# Waiting on irregular!
from lark.lexer import PatternRE
_operators = ['IN', 'NOT_IN', 'AND', 'OR']
def _edit_terminals(t):
    if t.name in _operators:
        t.pattern = PatternRE('%s(?!\w)' % t.pattern.value)

parser = Lark.open(
    'preql.lark',
    rel_to=__file__,
    parser='lalr',
    postlex=Postlexer(),
    start=['stmts', 'expr'],
    maybe_placeholders=True,
    propagate_positions=True,
    cache=True,
    edit_terminals=_edit_terminals,
    # transformer=T()
)


def terminal_desc(name):
    if name == '_NL':
        return "<NEWLINE>"
    p = parser.get_terminal(name).pattern
    if p.type == 'str':
        return p.value
    return '<%s>' % name

def terminal_list_desc(term_list):
    return [terminal_desc(x) for x in term_list if x != 'MARKER']

def parse_stmts(s, source_file, wrap_syntax_error=True):
    try:
        tree = parser.parse(s+"\n", start="stmts")
    except UnexpectedInput as e:
        if not wrap_syntax_error:
            raise

        assert isinstance(source_file, (str, Path)), source_file

        pos =  TextPos(e.pos_in_stream, e.line, e.column)
        ref = TextReference(s, str(source_file), TextRange(pos, pos))
        if isinstance(e, UnexpectedToken):
            if e.token.type == '$END':
                msg = "Code ended unexpectedly"
                ref = TextReference(s, str(source_file), TextRange(pos, TextPos(len(s), -1 ,-1)))
            else:
                msg = "Unexpected token: %r" % e.token.value

            expected = e.accepts or e.expected
            if expected and len(expected) < 5:
                accepts = terminal_list_desc(expected)
                msg += '. Expected: %s' % ' or '.join(accepts)
        else:
            msg = "Unexpected character: %r" % s[e.pos_in_stream]

        raise pql_SyntaxError(ref, msg)

    return TreeToAst(code_ref=(s, source_file)).transform(tree)
