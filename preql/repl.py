from time import time

import rich.console
import rich.markup

### XXX Fix for Python 3.8 bug (https://github.com/prompt-toolkit/python-prompt-toolkit/issues/1023)
import asyncio
import selectors
selector = selectors.SelectSelector()
loop = asyncio.SelectorEventLoop(selector)
asyncio.set_event_loop(loop)
### XXX End of fix

from pygments.lexers.go import GoLexer
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import Condition
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text.html import HTML, html_escape

from . import __version__
from . import pql_objects as objects
from .utils import memoize
from .api import table_more
from .exceptions import Signal, ExitInterp, pql_SyntaxError
from .pql_types import Object
from .parser import parse_stmts
from .loggers import repl_log
from . import settings
from .autocomplete import autocomplete



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
        context, fragment = last_word(document.text_before_cursor)

        if not settings.autocomplete:
            return

        context = context.rstrip()
        open_complete = context and context[-1] in '.{['
        if not fragment and not open_complete:
            return

        assert open_complete or is_name(fragment[-1]), fragment

        try:
            all_vars = dict(autocomplete(self.state, context))
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
                    display=HTML('<b>%s</b>%s: <blue>%s</blue>' % (a, b, html_escape(t))),
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
            repl_log.warn(e)

    return True


from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Number, Operator, Generic
from prompt_toolkit.styles.pygments import style_from_pygments_cls

class PreqlStyle(Style):
    default_style = ""
    styles = {
        Generic:                'ansigray',
        Comment:                'italic #888',
        Keyword:                'bold #005',
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

    console = rich.console.Console()
    console.print(f"[purple]Preql {__version__} interactive prompt. Type help() for help[/purple]")

    try:
        session = PromptSession(
            style=style_from_pygments_cls(PreqlStyle),
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
                    if code == '.':
                        console.print(table_more(p.interp.state), overflow='ellipsis')
                        continue

                    # Evaluate (Really just compile)
                    res = p._run_code(code, '<repl>')

                    # Print
                    if res is not None and res is not objects.null:
                        assert isinstance(res, Object), (res, type(res))

                        if save_last:
                            p.interp.set_var(save_last, res)

                        res = res.repr(p.interp.state)
                        # repl_log.info(res)
                        if isinstance(res, str):
                            if len(res) > 200:
                                res = res[:100] + "..." + res[-100:]    # smarter limit?
                            res = rich.markup.escape(res)
                        console.print(res, overflow='ellipsis')

                except Signal as s:
                    for is_rich, line in s.get_rich_lines():
                        if is_rich:
                            rich.print(line)
                        else:
                            print(line)
                    # repl_log.error(s)
                    # p.interp.set_var('_e', objects.ExceptionInstance(e))
                    continue
                except ExitInterp as e:
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

