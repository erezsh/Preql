from typing import Optional, List

from lark.exceptions import GrammarError

from .utils import dataclass, TextReference
from .base import Object

from .context import context

@dataclass
class Signal(Object, Exception):
    type: object    # Type
    text_refs: List[Optional[TextReference]]    # XXX must it be optional?
    message: Optional[str]

    @classmethod
    def make(cls, type, ast, message):
        ast_ref = getattr(ast, 'text_ref', None)
        try:
            refs = context.state.stacktrace+([ast_ref] if ast_ref else [])
        except AttributeError:
            refs = []
        return cls(type, refs, message)

    # def __str__(self):
    #     "Returns the exception with a traceback"
    #     texts = ['Exception traceback:\n']
    #     texts += [ref.get_pinpoint_text() if ref else '  ~~~ ???\n' for ref in self.text_refs]
    #     texts += [
    #         '[%s] %s\n' % (self.type, self.message)
    #     ]
    #     return ''.join(texts)

    def repr(self):
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
