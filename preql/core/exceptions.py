from typing import Optional, List

from lark.exceptions import GrammarError

from preql.utils import dataclass, TextReference
from preql.context import context

from .base import Object


@dataclass
class Signal(Object, Exception):
    type: object    # Type
    text_refs: List[Optional[TextReference]]    # XXX must it be optional?
    message: Optional[str]
    orig_exc: Optional[Exception] = None

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

    def __str__(self):
        return self.repr()

    def clean_copy(self):
        "Creates a copy of the object, without the attached stacktrace"
        s = Signal(self.type, self.text_refs, self.message, orig_exc=self)
        return s


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
