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




class Dataclass:
    def __post_init__(self):
        if not hasattr(self, '__annotations__'):
            return
        for name, type_ in self.__annotations__.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, type_):
                raise TypeError(f"[{self.__class__.__name__}] Attribute {name} expected value of type {type_}, instead got {value}")
            # assert value is None or isinstance(value, type_), (name, value, type_)



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