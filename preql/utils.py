from collections import deque
from dataclasses import dataclass

class FzSet(frozenset):
    def __repr__(self):
        return '{%s}' % ', '.join(map(repr, self))


def classify_bool(seq, pred):
    true_elems = []
    false_elems = []

    for elem in seq:
        if pred(elem):
            true_elems.append(elem)
        else:
            false_elems.append(elem)

    return true_elems, false_elems

def classify(seq, key=None, value=None):
    d = {}
    for item in seq:
        k = key(item) if (key is not None) else item
        v = value(item) if (value is not None) else item
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
    return d

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





from contextlib import contextmanager





class Context(list):
    def get(self, name, default=KeyError):
        for d in self[::-1]:
            if name in d:
                return d[name]
        if default is KeyError:
            raise KeyError(name)
        return default


    @contextmanager
    def push(self, **kwds):
        x = len(self)
        self.append(kwds)
        try:
            yield
        finally:
            self.pop()
            assert x == len(self)



import inspect
def make_define_decorator(base_class, frozen=True):

    def _post_init(obj):
        if inspect.isclass(obj):
            return

        assert isinstance(obj, base_class)

        if hasattr(obj, '_init'):
            obj._init()

        if not hasattr(obj, '__annotations__'):
            return

        for name, type_ in obj.__annotations__.items():
            value = getattr(obj, name)
            if value is not None:
                if isinstance(type_, list):
                    assert isinstance(value, list)
                    elem_type ,= type_
                    for elem in value:
                        if not isinstance(elem, elem_type):
                            raise TypeError(f'{type(obj).__name__}.{name} expects type {type_}, instead got element {elem!r}')

                elif not isinstance(value, type_):
                    raise TypeError(f'{type(obj).__name__}.{name} expects type {type_}, instead got {value!r}')

    def decorator(cls):
        cls.__post_init__ = _post_init
        c = dataclass(cls, frozen=frozen)
        attrs = c.__dict__.copy()
        return type(cls.__name__, (c, base_class), attrs)

    return decorator
                                  

class Dataclass:
    def __post_init__(self):
        if not hasattr(self, '__annotations__'):
            return
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, type_):
                raise TypeError(f"[{self.__class__.__name__}] Attribute {name} expected value of type {type_}, instead got {value}")
            # assert value is None or isinstance(value, type_), (name, value, type_)

