from lark import Token, UnexpectedCharacters, UnexpectedToken, ParseError

from preql.loggers import ac_log
from preql.utils import bfs_all_unique, dsp
from preql.context import context

from .exceptions import Signal, ReturnSignal, pql_SyntaxError
from .compiler import AutocompleteSuggestions
from .evaluate import evaluate, resolve
from .state import ThreadState, set_var, use_scope
from . import pql_ast as ast
from . import pql_objects as objects
from .pql_types import T
from . import sql, parser


@dsp
def eval_autocomplete(x, go_inside):
    _res = evaluate(x)
    # assert isinstance(res, objects.Instance)

@dsp
def eval_autocomplete(cb: ast.Statement, go_inside):
    raise NotImplementedError(cb)


@dsp
def eval_autocomplete(t: ast.Try, go_inside):
    eval_autocomplete(t.try_, go_inside)
    catch_type = evaluate(t.catch_expr)
    scope = {t.catch_name: Signal(catch_type, [], '<unknown exception>')} if t.catch_name else {}
    with use_scope(scope):
        eval_autocomplete(t.catch_block, go_inside)

@dsp
def eval_autocomplete(a: ast.InsertRows, go_inside):
    eval_autocomplete(a.value, go_inside)
@dsp
def eval_autocomplete(a: ast.Assert, go_inside):
    eval_autocomplete(a.cond, go_inside)
@dsp
def eval_autocomplete(a: ast.Print, go_inside):
    eval_autocomplete(a.value, go_inside)

@dsp
def eval_autocomplete(x: ast.If, go_inside):
    eval_autocomplete(x.then, go_inside)
    if x.else_:
        eval_autocomplete(x.else_, go_inside)

@dsp
def eval_autocomplete(x: ast.SetValue, go_inside):
    value = evaluate( x.value)
    if isinstance(x.name, ast.Name):
        set_var(x.name.name, value)

@dsp
def eval_autocomplete(cb: ast.CodeBlock, go_inside):
    for s in cb.statements[:-1]:
        eval_autocomplete(s, False)

    for s in cb.statements[-1:]:
        eval_autocomplete(s, go_inside)

@dsp
def eval_autocomplete(td: ast.TableDefFromExpr, go_inside):
    expr = evaluate(td.expr)
    assert isinstance(td.name, str)
    set_var(td.name, expr)

@dsp
def eval_autocomplete(td: ast.TableDef, go_inside):
    t = resolve(td)
    n ,= t.options['name'].parts
    set_var(n, objects.TableInstance.make(sql.unknown, t, []))

@dsp
def eval_autocomplete(td: ast.StructDef, go_inside):
    t = resolve(td)
    set_var(t.name, t)

@dsp
def eval_autocomplete(fd: ast.FuncDef, go_inside):
    f = fd.userfunc
    assert isinstance(f, objects.UserFunction)

    try:
        if go_inside:
            with use_scope({p.name:objects.unknown for p in f.params}):
                try:
                    eval_autocomplete(f.expr, go_inside)
                except ReturnSignal:
                    pass

    finally:
        cb = ast.CodeBlock([ast.Return(objects.unknown)])
        set_var(f.name, f.replace(expr=cb))

@dsp
def eval_autocomplete(r: ast.Return, go_inside):
    # Same as _execute
    value = evaluate( r.value)
    raise ReturnSignal(value)



_closing_tokens = {
    'RSQB': ']',
    'RBRACE': '}',
    'RPAR': ')',
    '$END': '<END>',
    '_NL': '\n',
}

def _search_parser(parser):
    def expand(p):
        for choice in p.choices():
            if choice in _closing_tokens:
                t = Token(choice, _closing_tokens[choice], 1, 1, 1, 1, 2, 2)
                try:
                    new_p = p.feed_token(t)
                except ParseError:    # Illegal
                    pass
                else:
                    yield new_p

    for p in bfs_all_unique([parser], expand):
        if p.result:
            return p.result

def autocomplete_tree(parser):
    if not parser:
        return

    # No marker, no autocomplete
    if 'MARKER' not in parser.choices():
        return

    # Feed marker
    t = Token('MARKER', '<MARKER>', 1, 1, 1, 1, 2, 2)
    try:
        res = parser.feed_token(t)
    except ParseError:    # Could still fail
        return

    assert not res, res

    # Search nearest solution
    return _search_parser(parser.as_immutable())


KEYWORDS = 'table update delete new func try if else for throw catch print assert const in or and not one null false true return !in'.split()
KEYWORDS = {k:(100000, None) for k in KEYWORDS}

class AcState(ThreadState):
    def get_var(self, name):
        try:
            return super().get_var(name)
        except Signal as s:
            assert s.type <= T.NameError
            return objects.UnknownInstance()

    # def get_all_vars(self):
    #     all_vars = dict(self.get_var('__builtins__').namespace)
    #     all_vars.update( self.ns.get_all_vars() )
    #     return all_vars

    def get_all_vars_with_rank(self):
        all_vars = {k:(10000, v) for k, v in self.get_var('__builtins__').namespace.items()}
        all_vars.update( self.ns.get_all_vars_with_rank() )
        all_vars.update(KEYWORDS)
        return all_vars

    def replace(self, **kw):
        assert False

def _eval_autocomplete(ac_state, stmts):
    for stmt in stmts:
        try:
            eval_autocomplete(stmt, False)
        except Signal as e:
            ac_log.exception(e)

def autocomplete(state, code, source='<autocomplete>'):
    ac_state = AcState.clone(state)
    with context(state=ac_state):
        try:
            stmts = parser.parse_stmts(code, source, wrap_syntax_error=False)
        except UnexpectedCharacters as e:
            return {}
        except UnexpectedToken as e:
            tree = autocomplete_tree(e.interactive_parser)
            if tree:
                try:
                    stmts = parser.TreeToAst(code_ref=(code, source)).transform(tree)
                except pql_SyntaxError as e:
                    return {}

                _eval_autocomplete(ac_state, stmts[:-1])

                try:
                    eval_autocomplete(stmts[-1], True)
                except AutocompleteSuggestions as e:
                    ns ,= e.args
                    return ns
                except Signal as e:
                    ac_log.exception(e)

        else:
            _eval_autocomplete(ac_state, stmts)

    return ac_state.get_all_vars_with_rank()