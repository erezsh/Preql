from typing import Optional, List

from .utils import dataclass, TextReference



class InsufficientAccessLevel(Exception):
    pass
class DatabaseQueryError(Exception):
    pass


@dataclass
class PreqlError(Exception):
    text_refs: List[TextReference]

    @property
    def exc_name(self):
        n = type(self).__name__
        if n.startswith('pql_'):
            n = n[4:]
        return n

    def __str__(self):
        texts = ['Exception traceback:\n']
        texts += [ref.get_pinpoint_text() for ref in self.text_refs]
        texts += [
            '[%s] %s\n' % (self.exc_name, self.message)
        ]
        return ''.join(texts)

    @classmethod
    def make(cls, state, ast, *args):
        ast_ref = getattr(ast, 'text_ref', None)
        return cls(state.stacktrace+([ast_ref] if ast_ref else []), *args)




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

class pql_CompileError(PreqlError_Message):
    pass



@dataclass
class pql_ExitInterp(Exception):
    value: object

@dataclass
class ReturnSignal(Exception):
    value: object

