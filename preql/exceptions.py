from .utils import dataclass

class PreqlError(Exception): pass

@dataclass
class pql_NameNotFound(PreqlError):
    name: str
    meta: dict

@dataclass
class pql_TypeError(PreqlError):
    message: str

@dataclass
class pql_SyntaxError(PreqlError):
    message: str
