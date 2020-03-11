from typing import _GenericAlias as TypeBase, Any, Union, Callable
from functools import wraps
from operator import getitem

from runtype import dataclass

from . import settings
dataclass = dataclass(check_types=settings.debug)


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


def find_duplicate(seq, key=lambda x:x):
    "Returns the first duplicate item in given sequence, or None if not found"
    found = set()
    for i in seq:
        k = key(i)
        if k in found:
            return i
        found.add(k)


class _X:
    def __init__(self, path = None):
        self.path = path or []

    def __getattr__(self, attr):
        x = getattr, attr
        return type(self)(self.path+[x])

    def __getitem__(self, item):
        x = getitem, item
        return type(self)(self.path+[x])

    def __call__(self, obj):
        for f, p in self.path:
            obj = f(obj, p)
        return obj

X = _X()


import time
from contextlib import contextmanager
class Benchmark:
    def __init__(self):
        self.total = {}

    @contextmanager
    def measure(self, name):
        if name not in self.total:
            self.total[name] = 0

        start = time.time()
        try:
            yield
        finally:
            total = time.time() - start
            self.total[name] += total

    def measure_func(self, f):
        @wraps(f)
        def _f(*args, **kwargs):
            with benchmark.measure(f.__name__):
                return f(*args, **kwargs)
        return _f

    def reset(self):
        self.total = {}

    def print(self):
        scores = [(total, name) for name, total in self.total.items()]
        scores.sort(reverse=True)
        print('---')
        for total, name in scores:
            print('%.4f\t%s' % (total, name))

benchmark = Benchmark()