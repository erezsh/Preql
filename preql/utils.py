from typing import _GenericAlias as TypeBase, Any, Union
from dataclasses import dataclass as _dataclass

def _isinstance(a, b):
    try:
        return isinstance(a, b)
    except TypeError as e:
        raise TypeError(f"Bad arguments to isinstance: {a}, {b}") from e

def isa(obj, t):
    if t is Any or t == (Any,):
        return True
    elif _isinstance(t, TypeBase):
        if t.__origin__ is list:
            return all(isa(item, t.__args__) for item in obj)
        elif t.__origin__ is dict:
            kt, vt = t.__args__
            return all(isa(k, kt) and isa(v, vt) for k, v in obj.items())
        elif t.__origin__ is Union:
            return _isinstance(obj, t.__args__)
        assert False, t.__origin__
    return _isinstance(obj, t)


def __post_init(self):
    if not hasattr(self, '__dataclass_fields__'):
        return
    for name, field in self.__dataclass_fields__.items():
        value = getattr(self, name)
        if not isa(value, field.type):
            raise TypeError(f"[{type(self).__name__}] Attribute {name} expected value of type {field.type}, instead got {value!r}")

    if hasattr(self, '__created__'):
        self.__created__()



def dataclass(cls, frozen=True):
    cls.__post_init__ = __post_init
    return _dataclass(cls, frozen=frozen)


class SafeDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise KeyError("Attempted to override key '%s' in a SafeDict" % key)
        return dict.__setitem__(self, key, value)

    def update(self, *other_dicts):
        for other in other_dicts:
            for k, v in other.items():
                self[k] = v
        return self

def concat(*iters):
    return [elem for it in iters for elem in it]


def safezip(*args):
    assert len(set(map(len, args))) == 1
    return zip(*args)

def split_at_index(arr, idx):
    return arr[:idx], arr[idx:]