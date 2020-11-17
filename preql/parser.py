from ast import literal_eval
from pathlib import Path

from lark import Lark, Transformer, v_args, UnexpectedInput, UnexpectedToken, Token

from .utils import TextPos, TextRange, TextReference
from .exceptions import pql_SyntaxError
from . import pql_ast as ast
from . import pql_objects as objects

from .pql_types import T

class Str(str):
    def __new__(cls, value, text_ref=None):
        obj = str.__new__(cls, value)
        obj.text_ref = text_ref
        return obj

    def set_text_ref(self, text_ref):
        # Required for parity with Ast
        self.text_ref = text_ref


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
            assert c.text_ref.text is text
            assert c.text_ref.context is None
            c.text_ref.context = ref

    assert isinstance(source_file, (str, Path)), source_file
    return TextReference(text, str(source_file), ref)




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

from .compiler import guess_field_name  # XXX a little out of place


def _wrap_result(res, f, meta, children):
    if isinstance(res, (Str, ast.Ast)):
        ref = make_text_reference(*f.__self__.code_ref, meta, children)
        res.set_text_ref(ref)
    return res

def _args_wrapper(f, data, children, meta):
    "Create meta with 'code' from transformer"
    res = f(*children)
    return _wrap_result(res, f, meta, children)

def _args_wrapper_meta(f, data, children, meta):
    res = f(meta, *children)
    return _wrap_result(res, f, meta, children)

def _args_wrapper_list(f, data, children, meta):
    res = f(children)
    return _wrap_result(res, f, meta, children)

with_meta = v_args(wrapper=_args_wrapper_meta)
no_inline = v_args(wrapper=_args_wrapper_list)

def token_value(self, t):
    return Str(str(t))


@v_args(wrapper=_args_wrapper)
class TreeToAst(Transformer):
    def __init__(self, code_ref):
        self.code_ref = code_ref

    name = token_value

    def string(self, s):
        return ast.Const(T.string, _fix_escaping( s.value[1:-1]) )
    def long_string(self, s):
        return ast.Const(T.string, _fix_escaping( s.value[3:-3]) )

    @with_meta
    def pql_dict(self, meta, items):
        d = {}
        for item in items:
            name = item.name or guess_field_name(item.value)
            if name in d:
                raise pql_SyntaxError(meta, f"Dict key appearing more than once: {name}")
            d[name] = item.value

        return ast.Dict_(d)

    def int(self, i):
        return ast.Const(T.int, int(i))

    def float(self, f):
        return ast.Const(T.float, float(f))

    def null(self):
        return ast.Const(T.nulltype, None)
    def false(self):
        return ast.Const(T.bool, False)
    def true(self):
        return ast.Const(T.bool, True)

    @no_inline
    def pql_list(self, items):
        return ast.List_(T.list[T.any], items)

    @no_inline
    def as_list(self, args):
        return args

    # types
    def typemod(self, *args):
        return [t.value for t in args]
    def type(self, name, mods):
        # TODO pk
        return ast.Type(name, '?' in (mods or ''))

    add_op = token_value
    mul_op = token_value
    comp_op = token_value

    def compare(self, a, op, b):
        return ast.Compare(op, [a,b])

    def _arith_expr(self, a, op, b):
        return ast.Arith(op, [a,b])

    add_expr = _arith_expr
    term = _arith_expr
    power = _arith_expr

    and_test = no_inline(ast.And)
    or_test = no_inline(ast.Or)

    not_test = ast.Not

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

    def projection_grouped(self, table, keys, values):
        return ast.Projection(table, keys, True, values)

    def projection_grouped_nokeys(self, table, values):
        return ast.Projection(table, [], True, values)

    def projection_grouped_novalues(self, table, keys):
        return ast.Projection(table, keys, True, [])

    # Statements / Declarations
    param = objects.Param
    param_variadic = objects.ParamVariadic

    @with_meta
    def func_def(self, meta, name, params, expr):
        collector = None
        for i, p in enumerate(params):
            if isinstance(p, objects.ParamVariadic):
                if i != len(params)-1:
                    raise pql_SyntaxError(meta, f"A variadic parameter may only appear at the end of the function ({p.name})")

                collector = p
                params = params[:-1]

        return ast.FuncDef(objects.UserFunction(name, params, expr, collector))

    @with_meta
    def func_call(self, meta, func, args):
        for i, a in enumerate(args):
            if isinstance(a, ast.InlineStruct):
                if i != len(args)-1:
                    raise pql_SyntaxError(meta, f"An inlined struct must appear at the end of the function call ({a})")


        return ast.FuncCall(func, args)

    def set_value(self, lval, rval):
        if not isinstance(lval, (ast.Name, ast.Attr)):
            raise pql_SyntaxError(lval.text_ref, f"{lval.type} is not a valid l-value")
        return ast.SetValue(lval, rval)

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
    one = lambda self, nullable, expr: ast.One(expr, nullable is not None)

    def marker(self, _marker):
        return ast.Marker()

    def table_def_from_expr(self, const, name, table_expr):
        return ast.TableDefFromExpr(name, table_expr, const == 'const')

    codeblock = no_inline(ast.CodeBlock)
    ellipsis = no_inline(ast.Ellipsis)


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
# Waiting on interregular!
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