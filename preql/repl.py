import rich.console
import rich.markup

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
from .exceptions import PreqlError, pql_ExitInterp, pql_SyntaxError
from .pql_types import Object
from .parser import parse_stmts
from .loggers import ac_log, repl_log
from . import settings
from preql.autocomplete import autocomplete
from .utils import memoize

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


def is_name(s):
    return s.isalnum() or s in ('_', '!')

def last_word(s):
    if not s:
        return '', ''
    i = len(s)
    while i and is_name(s[i-1]):
        i -= 1
    if i < len(s) and s[i] == '!' :
        i += 1  # hack to support ... !var and !in
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


@memoize
def _code_is_valid(code):
    if code:
        try:
            s = parse_stmts(code, '<repl>')
        except pql_SyntaxError as e:
            return False
        except Exception as e:
            repl_log.warn(e)

    return True


def start_repl(p, prompt=' >> '):
    repl_log.info("Welcome to the Preql REPL. Type help() for help")
    save_last = '_'   # XXX A little hacky

    p.interp.state.fmt = 'rich' # TODO proper api

    console = rich.console.Console()

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
            return not _code_is_valid(text)

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
                        # repl_log.info(res)
                        if isinstance(res, str):
                            res = rich.markup.escape(res)
                        console.print(res, overflow='ellipsis')

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