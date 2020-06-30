from collections import deque

from lark import Token, UnexpectedCharacters, UnexpectedToken, Tree

from .exceptions import ReturnSignal, pql_NameNotFound, PreqlError
from .loggers import ac_log
from . import pql_ast as ast
from . import pql_objects as objects
from .utils import bfs
from .parser import parse_stmts, TreeToAst
from .pql_types import T

from .interp_common import State, dy, new_value_instance, evaluate
from .evaluate import evaluate, execute, resolve
from .compiler import AutocompleteSuggestions
from . import sql


@dy
def eval_autocomplete(state, x, go_inside):
    res = evaluate(state, x)
    # assert isinstance(res, objects.Instance)

@dy
def eval_autocomplete(state, cb: ast.Statement, go_inside):
    raise NotImplementedError(cb)

@dy
def eval_autocomplete(state, x: ast.If, go_inside):
    eval_autocomplete(state, x.then, go_inside)
    if x.else_:
        eval_autocomplete(state, x.else_, go_inside)

@dy
def eval_autocomplete(state, x: ast.SetValue, go_inside):
    value = evaluate(state, x.value)
    if isinstance(x.name, ast.Name):
        state.set_var(x.name.name, value)

@dy
def eval_autocomplete(state, cb: ast.CodeBlock, go_inside):
    for s in cb.statements[:-1]:
        eval_autocomplete(state, s, False)

    for s in cb.statements[-1:]:
        eval_autocomplete(state, s, go_inside)

@dy
def eval_autocomplete(state, td: ast.TableDef, go_inside):
    t = resolve(state, td)
    state.set_var(t.options['name'], objects.TableInstance.make(sql.unknown, t, []))

@dy
def eval_autocomplete(state, fd: ast.FuncDef, go_inside):
    f = fd.userfunc
    assert isinstance(f, objects.UserFunction)

    try:
        if go_inside:
            with state.use_scope({p.name:T.unknown for p in f.params}):
                try:
                    eval_autocomplete(state, f.expr, go_inside)
                except ReturnSignal:
                    pass

    finally:
        cb = ast.CodeBlock(None, [ast.Return(None, objects.unknown)])
        state.set_var(f.name, f.replace(expr=cb))

@dy
def eval_autocomplete(state: State, r: ast.Return, go_inside):
    # Same as _execute
    value = evaluate(state, r.value)
    raise ReturnSignal(value)



_closing_tokens = {
    'RSQB': ']',
    'RBRACE': '}',
    'RPAR': ')',
    '$END': '<END>',
    '_NL': '\n',
}

def _search_puppet(puppet):
    def expand(p):
        for choice in p.choices():
            if choice in _closing_tokens:
                t = Token(choice, _closing_tokens[choice], 1, 1, 1, 1, 2, 2)
                new_p = p.copy()
                try:
                    res = new_p.feed_token(t)
                except KeyError:    # Illegal
                    pass
                else:
                    yield new_p

    for p in bfs([puppet], expand):
        if p.result:
            return p.result

def autocomplete_tree(puppet):
    if not puppet:
        return

    # No marker, no autocomplete
    if 'MARKER' not in puppet.choices():
        return

    # Feed marker
    t = Token('MARKER', '<MARKER>', 1, 1, 1, 1, 2, 2)
    try:
        res = puppet.feed_token(t)
    except KeyError:    # Could still fail
        return

    assert not res

    # Search nearest solution
    return _search_puppet(puppet)


from .interp_common import State
class AcState(State):
    def get_var(self, name):
        try:
            return self.ns.get_var(self, name)
        except pql_NameNotFound:
            return objects.UnknownInstance()

    def replace(self, **kw):
        assert False

def autocomplete(state, code, source='<autocomplete>'):
    try:
        parse_stmts(code, source, wrap_syntax_error=False)
    except UnexpectedCharacters as e:
        return {}
    except UnexpectedToken as e:
            tree = autocomplete_tree(e.puppet)
            if tree:
                stmts = TreeToAst(code_ref=(code, source)).transform(tree)
                ac_state = AcState.clone(state)

                for stmt in stmts[:-1]:
                    try:
                        eval_autocomplete(ac_state, stmt, False)
                    except PreqlError as e:
                        ac_log.exception(e)

                try:
                    eval_autocomplete(ac_state, stmts[-1], True)
                except AutocompleteSuggestions as e:
                    ns ,= e.args
                    return ns
                except PreqlError as e:
                    ac_log.exception(e)

    ns = state.ns.get_all_vars()
    return ns
