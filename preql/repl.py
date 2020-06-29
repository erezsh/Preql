from time import time
import sys

### XXX Fix for Python 3.8 bug (https://github.com/prompt-toolkit/python-prompt-toolkit/issues/1023)
import asyncio
import selectors
selector = selectors.SelectSelector()
loop = asyncio.SelectorEventLoop(selector)
asyncio.set_event_loop(loop)
### XXX End of fix

from . import Preql
from . import pql_objects as objects
from . import pql_ast as ast
from .api import TablePromise
from .exceptions import PreqlError, pql_ExitInterp, pql_SyntaxError_PrematureEnd, pql_SyntaxError, pql_NameNotFound
from .pql_types import Object
from .parser import parse_stmts
from .loggers import ac_log, repl_log
from . import settings


# class RowWrapper:
#     def __init__(self, row):
#         self._row = row

#     def __repr__(self):
#         return self._row.repr()

#     def __getitem__(self, item):
#         return self._row.getattr(item)

#     def __getattr__(self, attr):
#         return self[attr]

#     def __iter__(self):
#         return iter(self._row)

#     def __getstate__(self):
#         return self._row
#     def __setstate__(self, x):
#         self._row = x


# class TableWrapper:
#     def __init__(self, pql_table, interp):
#         self._pql_table = pql_table
#         self._interp = interp

#     def __repr__(self):
#         return self._pql_table.repr(self._interp)

#     def json(self):
#         return [row.attrs for row in self._query()]

#     def _query(self):
#         return self._pql_table.query(self._interp, None)

#     def __iter__(self):
#         return (RowWrapper(row) for row in self._query())

#     def __len__(self):
#         return self._pql_table.count(self._interp).value


from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from pygments.lexers.python import Python3Lexer
from pygments.lexers.go import GoLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import Condition
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.application.current import get_app

from prompt_toolkit.completion import WordCompleter

KEYWORDS = 'table update delete new func try if else for throw catch print assert const in or and not one null false true return !in'.split()
# word_completer = WordCompleter(KEYWORDS)
KEYWORDS = {k:None for k in KEYWORDS}

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text.html import HTML, html_escape


_closing_tokens = {
    'RSQB': ']',
    'RBRACE': '}',
    'RPAR': ')',
    '$END': '<END>',
    '_NL': '\n',
}

from lark import Token, UnexpectedCharacters, UnexpectedToken, Tree
from preql.parser import parse_stmts, TreeToAst
from preql.evaluate import evaluate, execute
from preql.autocomplete import eval_autocomplete
from preql.compiler import AutocompleteSuggestions
from preql.exceptions import PreqlError

from collections import deque
def bfs(initial, expand):
    open_q = deque(list(initial))
    visited = set(open_q)
    while open_q:
        node = open_q.popleft()
        yield node
        for next_node in expand(node):
            if next_node not in visited:
                visited.add(next_node)
                open_q.append(next_node)

def just_tree_data(l):
    return [x.data for x in l if isinstance(x, Tree)]

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
    choices = puppet.choices()
    if '_MARKER' not in choices:
        return

    # Feed marker
    t = Token('_MARKER', '<MARKER>', 1, 1, 1, 1, 2, 2)
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
    except UnexpectedCharacters:
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
                    # TODO autocomplete_execute
                    # execute(state, stmt)
                    eval_autocomplete(ac_state, stmts[-1], True)
                except AutocompleteSuggestions as e:
                    ns = e.args[0]
                    return ns
                except PreqlError as e:
                    ac_log.exception(e)

    ns = state.ns.get_all_vars()
    return ns

def is_name(s):
    return s.isalnum() or s in ('_', '!')

def last_word(s):
    if not s:
        return '', ''
    i = len(s)
    while i and is_name(s[i-1]):
        i -= 1
    return s[:i], s[i:]


class Autocompleter(Completer):
    def __init__(self, state):
        self.state = state

    def get_completions(self, document, complete_event):
        context, fragment = last_word(document.text_before_cursor)

        if not settings.autocomplete:
            return

        if not fragment:
            return

        assert is_name(fragment[-1])

        all_vars = autocomplete(self.state, context)
        all_vars.update(KEYWORDS)

        for k,v in all_vars.items():
            if k.startswith(fragment):
                a, b = k[:len(fragment)], k[len(fragment):]
                if v is None:
                    t = "<keyword>"
                else:
                    try:
                        t = v.type
                    except AttributeError:
                        t = type(v)

                yield Completion(
                    b, start_position=0,
                    display=HTML('<b>%s</b>%s: <blue>%s</blue>' % (a, b, html_escape(t))),
                    )

class MyValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text.strip():
            return

        try:
            s = parse_stmts(text, '<repl>')
        except pql_SyntaxError as e:
            text_ref, message = e.args
            if text_ref:
                pos = text_ref[-1].ref.end.char_index
                if pos < len(text):
                    raise ValidationError(message=message, cursor_position=pos)
        # except Exception as e:
            # raise ValidationError(message=e.args[0], cursor_position=0)
            # pass

from prompt_toolkit.key_binding import KeyBindings
kb = KeyBindings()
@kb.add('c-space')
def _(event):
    " Initialize autocompletion, or select the next completion. "
    buff = event.app.current_buffer
    if buff.complete_state:
        buff.complete_next()
    else:
        buff.start_completion(select_first=False)


def start_repl(p, prompt=' >> '):
    repl_log.info("Welcome to the Preql REPL. Type help() for help")
    save_last = '_'   # XXX A little hacky

    try:
        session = PromptSession(
            lexer=PygmentsLexer(GoLexer),
            completer=Autocompleter(p.interp.state),
            # key_bindings=kb
            validator=MyValidator(),
        )

        @Condition
        def multiline_filter():
            text = get_app().layout.get_buffer_by_name('DEFAULT_BUFFER').text
            if text:
                try:
                    s = parse_stmts(text, '<repl>')
                except pql_SyntaxError as e:
                    return True
                except Exception as e:
                    repl_log.warn(e)
                    return False

            return False

        while True:
            # Read
            try:
                code = session.prompt(prompt, multiline=multiline_filter)
                if not code.strip():
                    continue

                start_time = time()
                try:
                    # Evaluate (Really just compile)
                    res = p.run_code(code, '<repl>')

                    # Print
                    if res is not None and res is not objects.null:
                        assert isinstance(res, Object), (res, type(res))

                        if save_last:
                            p.interp.set_var(save_last, res)

                        res = res.repr(p.interp.state)
                        repl_log.info(res)

                except PreqlError as e:
                    repl_log.error(e)
                    # p.interp.set_var('_e', objects.ExceptionInstance(e))
                    continue
                except pql_ExitInterp as e:
                    return
                except Exception as e:
                    repl_log.exception(e)
                    raise
                    # continue

                duration = time() - start_time
                if duration > 1:
                    repl_log.info("(Query took %.2f seconds)" % duration)

            except KeyboardInterrupt:
                repl_log.info("Interrupted (Ctrl+C)")



    except (KeyboardInterrupt, EOFError):
        repl_log.info('Exiting Preql interaction')


def main(script=None):
    # p = Preql(db)
    p = Preql()
    if script:
        p.load(script)
    start_repl(p)

if __name__ == '__main__':
    main(*sys.argv[1:])