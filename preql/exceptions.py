from typing import Optional, List

from lark.exceptions import GrammarError

from .utils import dataclass, TextReference
from .base import Object

@dataclass
class Signal(Object, Exception):
    type: object    # Type
    text_refs: List[Optional[TextReference]]    # XXX must it be optional?
    message: Optional[str]

    @classmethod
    def make(cls, type, state, ast, message):
        ast_ref = getattr(ast, 'text_ref', None)
        refs = state.stacktrace+([ast_ref] if ast_ref else [])
        return cls(type, refs, message)

    def __str__(self):
        "Returns the exception with a traceback"
        texts = ['Exception traceback:\n']
        texts += [ref.get_pinpoint_text() if ref else '  ~~~ ???\n' for ref in self.text_refs]
        texts += [
            '[%s] %s\n' % (self.type, self.message)
        ]
        return ''.join(texts)

    def get_rich_lines(self):
        "Yields colorful styled lines to print by the ``rich`` library"
        yield True, '[bold]Exception traceback:[/bold]'
        for ref in self.text_refs:
            yield from ref.get_pinpoint_text(rich=True) if ref else ['???']
        # yield True, '[red]%s[/red]: %s' % (self.exc_name, self.message)
        yield False, '%s: %s' % (self.type, self.message)

    def repr(self, state):
        return f'{self.type}("{self.message}")'


@dataclass
class pql_SyntaxError(GrammarError):
    text_ref: TextReference
    message: str

@dataclass
class ExitInterp(Exception):
    value: object

@dataclass
class ReturnSignal(Exception):
    value: object


@dataclass
class pql_AttributeError(Exception):
    message: str

class InsufficientAccessLevel(Exception):
    pass
class DatabaseQueryError(Exception):
    pass
