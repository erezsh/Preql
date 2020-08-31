from typing import Optional, List

from .utils import dataclass, TextReference
from .base import Object



class InsufficientAccessLevel(Exception):
    pass
class DatabaseQueryError(Exception):
    pass


@dataclass
class PreqlError(Object, Exception):
    text_refs: List[Optional[TextReference]]    # XXX must it be optional?

    @property
    def exc_name(self):
        n = type(self).__name__
        if n.startswith('pql_'):
            n = n[4:]
        return n

    def __str__(self):
        texts = ['Exception traceback:\n']
        texts += [ref.get_pinpoint_text() if ref else '  ~~~ ???\n' for ref in self.text_refs]
        texts += [
            '[%s] %s\n' % (self.exc_name, self.message)
        ]
        return ''.join(texts)

    def get_rich_lines(self):
        yield True, '[bold]Exception traceback:[/bold]'
        for ref in self.text_refs:
            yield from ref.get_pinpoint_text(rich=True) if ref else ['???']
        # yield True, '[red]%s[/red]: %s' % (self.exc_name, self.message)
        yield False, '%s: %s' % (self.exc_name, self.message)

    @classmethod
    def make(cls, state, ast, *args):
        ast_ref = getattr(ast, 'text_ref', None)
        return cls(state.stacktrace+([ast_ref] if ast_ref else []), *args)


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
        texts = ['Exception traceback:\n']
        texts += [ref.get_pinpoint_text() if ref else '  ~~~ ???\n' for ref in self.text_refs]
        texts += [
            '[%s] %s\n' % (self.type, self.message)
        ]
        return ''.join(texts)


@dataclass
class pql_NameNotFound(PreqlError):
    name: str

    @property
    def message(self):
        return "Name not found: '%s'" % self.name

@dataclass
class PreqlError_Message(PreqlError):
    message: str


class pql_TypeError(PreqlError_Message):
    pass

class pql_ImportError(PreqlError_Message):
    pass

class pql_AttributeError(PreqlError_Message):
    pass

class pql_ValueError(PreqlError_Message):
    pass

class pql_SyntaxError(PreqlError_Message):
    pass

class pql_SyntaxError_PrematureEnd(pql_SyntaxError):
    pass

class pql_JoinError(PreqlError_Message):
    pass

class pql_NotImplementedError(PreqlError_Message):
    pass

class pql_DatabaseQueryError(PreqlError_Message):
    pass

class pql_DatabaseConnectError(PreqlError_Message):
    pass

class pql_CompileError(PreqlError_Message):
    pass

class pql_AssertionError(PreqlError_Message):
    pass



@dataclass
class ExitInterp(Exception):
    value: object

@dataclass
class ReturnSignal(Exception):
    value: object

