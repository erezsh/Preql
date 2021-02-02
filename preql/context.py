import threading
from contextlib import contextmanager

class Context(threading.local):
    def __init__(self):
        self._ctx = [{}]

    def __getattr__(self, name):
        for scope in reversed(self._ctx):
            if name in scope:
                return scope[name]

        raise AttributeError(name)

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    @contextmanager
    def __call__(self, **attrs):
        self._ctx.append(attrs)
        try:
            yield
        finally:
            _d = self._ctx.pop()
            assert attrs is _d



def test_threading():
    import time, random

    context = Context()

    def f(i):
        with context(i=i):
            g(i)

    def g(i):
        assert context.i == i
        time.sleep(random.random())
        assert context.i == i
        print(i, end=', ')


    for i in range(100):
        t = threading.Thread(target=f, args=(i,))
        t.start()


# test_threading()

context = Context()
