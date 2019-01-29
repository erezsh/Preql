from pathlib import Path

from .interpreter import Interpreter

class SqlEngine:
    pass

class SqliteEngine(SqlEngine):
    def __init__(self, filename=None):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')

    def query(self, sql):
        dargs = {}
        c = self._conn.cursor()
        for i, s in enumerate(sql.split('\n')):
            print(' ...' if i else 'SQL>', s)
        c.execute(sql, dargs)
        return c.fetchall()


class Interface:
    def __init__(self):
        self.interp = Interpreter(SqliteEngine())

    def __call__(self, q, *args, **kw):
        return self.interp.execute(q, *args, **kw)

    def __getattr__(self, fname):
        def delegate(*args, **kw):
            assert not kw
            return self.interp.call_func(fname, args)
        return delegate

    def __getitem__(self, pq):
        sql = self._compiler.compile_query(pq)
        return self._query(sql, [])

    def load(self, fn, rel_to=None):
        """Load content filename as Preql code

        If rel_to is provided, the function will find the filename in relation to it.
        """
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self(f.read())


def test1():
    i = Interface()
    i.load('simple1.pql', rel_to=__file__)
    print(i.english())
    print(i.by_country('Israel'))
    print(i.english2())

def test2():
    i = Interface()
    i.load('simple2.pql', rel_to=__file__)
    # print(i.english_speakers())
    # print(i.person_and_language())
    # print(i.from_my_country())
    # print(i.population_count())
    # print(i.citizens_list())
    # print(i.explicit_join2())
    print(i.explicit_join())

def test3():
    a = open("preql/tree.pql").read()
    i = Interface()
    i(a)
    lion = i.lion()
    print(lion)
    print(i.up(lion['id']))

    # i.up() - requires join aliases!
    # i.animals() - Requires advanced type system

# print('---------')
# # test1()
# print('---------')
# test2()