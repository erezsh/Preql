from pathlib import Path
from time import time

### XXX Fix for Python 3.8 bug (https://github.com/prompt-toolkit/python-prompt-toolkit/issues/1023)
import asyncio
import selectors
selector = selectors.SelectSelector()
loop = asyncio.SelectorEventLoop(selector)
asyncio.set_event_loop(loop)
### XXX End of fix

from pygments.lexers.go import GoLexer
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Number, Operator, Generic
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import Condition
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text.html import HTML, html_escape
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.output.color_depth import ColorDepth

from . import __version__
from . import settings
from .utils import memoize
from .loggers import repl_log
from .context import context

from .core.exceptions import Signal, ExitInterp, pql_SyntaxError
from .core.autocomplete import autocomplete
from .core.parser import parse_stmts
from .core import pql_objects as objects
from .core.display import table_more
from .core.pql_types import Object
from .core.pql_types import T


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

    def _get_completions(self, document):
        previous_text, fragment = last_word(document.text_before_cursor)

        if not settings.autocomplete:
            return

        previous_text = previous_text.rstrip()
        open_complete = previous_text and previous_text[-1] in '.{['
        if not fragment and not open_complete:
            return

        assert open_complete or is_name(fragment[-1]), fragment

        try:
            all_vars = dict(autocomplete(self.state, previous_text))
        except:
            if settings.debug:
                raise
            all_vars = {}

        # all_vars.update(KEYWORDS)
        assert all(isinstance(v, tuple) for v in all_vars.values())
        all_vars = list(all_vars.items())
        all_vars.sort(key=lambda item: (item[1][0], item[0]))

        for k, (_rank, v) in all_vars:
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
                    display=HTML('<b>%s</b>%s<ansibrightblack> : %s</ansibrightblack>' % (a, b, html_escape(t))),
                    style='bg:ansigray fg:black',
                    selected_style="fg:black bg:ansibrightyellow",
                    )

    def get_completions(self, document, complete_event):
        return self._get_completions(document)

class MyValidator(Validator):
    def validate(self, document):
        text = document.text
        if not text.strip():
            return
        if text == '.':
            return

        try:
            parse_stmts(text, '<repl>')
        except pql_SyntaxError as e:
            if e.text_ref:
                pos = e.text_ref.ref.end.char_index
                # if pos <= len(text):
                raise ValidationError(message=e.message, cursor_position=pos)
        # except Exception as e:
            # raise ValidationError(message=e.args[0], cursor_position=0)
            # pass

# from prompt_toolkit.key_binding import KeyBindings
# kb = KeyBindings()
# @kb.add('c-space')
# def _(event):
#     " Initialize autocompletion, or select the next completion. "
#     buff = event.app.current_buffer
#     if buff.complete_state:
#         buff.complete_next()
#     else:
#         buff.start_completion(select_first=False)


@memoize
def _code_is_valid(code):
    if code == '.':
        return True
    if code:
        try:
            parse_stmts(code, '<repl>')
        except pql_SyntaxError as e:
            return False
        except Exception as e:
            repl_log.warning(e)

    return True


class PreqlStyle(Style):
    default_style = ""
    styles = {
        Generic:                'ansigray',
        Comment:                'italic #888',
        Keyword:                'bold #00f',
        Name:                   '#fff',
        Name.Function:          'bold #8f8',
        Name.Class:             'bold #0f0',
        String:                 'ansigreen',
        Number:                 'ansicyan',
        Operator:               'ansigray',
        Error:                  'bg:ansired ansigray',
    }


def start_repl(p, prompt=' >> '):
    save_last = '_'   # XXX A little hacky

    p.set_output_format('rich')

    display = p._display
    interp = p._interp
    console = display.console
    console.print(f"[purple]Preql {__version__} interactive prompt. Type help() for help[/purple]")

    try:
        session = PromptSession(
            style=style_from_pygments_cls(PreqlStyle),
            lexer=PygmentsLexer(GoLexer),
            completer=Autocompleter(interp.state),
            # key_bindings=kb
            validator=MyValidator(),
            history=FileHistory(str(Path.home() / '.preql_history')),
            auto_suggest=AutoSuggestFromHistory(),
            color_depth=ColorDepth.TRUE_COLOR,
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
                    if code == '.':
                        with context(state=p._interp.state):
                            console.print(table_more(), overflow='ellipsis')
                        continue

                    # Evaluate (Really just compile)
                    res = p._run_code(code, '<repl>')

                    # Print
                    if res is not None and res is not objects.null:
                        assert isinstance(res, Object), (res, type(res))

                        if save_last:
                            p._interp.set_var(save_last, res)

                        with context(state=p._interp.state):
                            res_repr = res.repr()

                        # repl_log.info(res)
                        if isinstance(res_repr, str) and res.type == T.string:  # Not text
                            if len(res_repr) > 200:
                                res_repr = res_repr[:100] + "..." + res_repr[-100:]    # smarter limit?
                        display.print(res_repr)

                except Signal as s:
                    display.print_exception(s)
                    # repl_log.error(s)
                    # p.interp.set_var('_e', objects.ExceptionInstance(e))
                    continue
                except ExitInterp as e:
                    return e.value
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
