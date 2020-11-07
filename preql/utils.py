import time
import re
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from typing import Optional
from functools import wraps
from operator import getitem
import dataclasses

import runtype


from . import settings

mut_dataclass = runtype.dataclass(check_types=settings.typecheck, frozen=False)
dataclass = runtype.dataclass(check_types=settings.typecheck)
dy = runtype.Dispatch()

def field_list():
    return dataclasses.field(default_factory=list)

class SafeDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            if value is self[key]:
                return  # Nothing to do
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


def classify_bool(seq, pred):
    true_elems = []
    false_elems = []

    for elem in seq:
        if pred(elem):
            true_elems.append(elem)
        else:
            false_elems.append(elem)

    return true_elems, false_elems

benchmark = Benchmark()


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



@dataclass
class TextPos:
    char_index: int
    line: int
    column: int

@dataclass
class TextRange:
    start: TextPos
    end: TextPos


def expand_tab(s):
    return s.replace('\t', '    ')

@mut_dataclass
class TextReference:
    text: str
    source_file: str
    ref: TextRange
    context: Optional[TextRange] = None

    def get_surrounding_line(self, span):
        pos = self.ref.start.char_index
        start = max(pos - span, 0)
        end = pos + span
        text_before = self.text[start:pos].rsplit('\n', 1)[-1]
        text_after = self.text[pos:end].split('\n', 1)[0]
        return expand_tab(text_before), expand_tab(text_after)

    def get_pinpoint_text(self, span=80, rich=False):
        text_before, text_after = self.get_surrounding_line(span)

        MARK_CHAR = '-'
        mark_before = mark_after = 0
        if self.context:
            pos = self.ref.start.char_index
            mark_before = max(0, min(len(text_before), pos - self.context.start.char_index))
            mark_after = max(0, min(len(text_after), self.context.end.char_index - pos - 1))
            assert mark_before >= 0 and mark_after >= 0

        source = Path(self.source_file)

        if rich:
            start = self.ref.start
            return [
                (True, f"  [red]~~~[/red] At '{source.name}' line {start.line}, column {start.column}"),
                (False, text_before + text_after),
                (False, ' ' * (len(text_before)-mark_before) + MARK_CHAR*mark_before + '^' + MARK_CHAR*mark_after),
            ]

        res = [
            "  ~~~ At '%s' line %d, column %d:\n" % (source.name, self.ref.start.line, self.ref.start.column),
            text_before, text_after, '\n',
            ' ' * (len(text_before)-mark_before), MARK_CHAR*mark_before, '^', MARK_CHAR*mark_after, '\n'
        ]

        return ''.join(res)

    def __str__(self):
        return '<text-ref>'
    def __repr__(self):
        return '<text-ref>'

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

def bfs_all_unique(initial, expand):
    open_q = deque(list(initial))
    while open_q:
        node = open_q.popleft()
        yield node
        open_q += expand(node)


def memoize(f, memo=None):
    if memo is None:
        memo = {}

    @wraps(f)
    def inner(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]

    return inner


@listgen
def re_split(r, s):
    offset = 0
    for m in re.finditer(r, s):
        yield None, s[offset:m.start()]
        yield m, s[m.start():m.end()]
        offset = m.end()
    yield None, s[offset:]
