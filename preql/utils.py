from typing import _GenericAlias as TypeBase, Any, Union, Callable
from functools import wraps

from runtype import dataclass


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

def merge_dicts(dicts):
    return SafeDict().update(*dicts)

def concat(*iters):
    return [elem for it in iters for elem in it]
def concat_for(iters):
    return [elem for it in iters for elem in it]


def safezip(*args):
    assert len(set(map(len, args))) == 1
    return zip(*args)

def split_at_index(arr, idx):
    return arr[:idx], arr[idx:]


def listgen(f):
    @wraps(f)
    def _f(*args, **kwargs):
        return list(f(*args, **kwargs))
    return _f

