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


def test1():
    a = open("preql/simple1.pql").read()
    i = Interface()
    i(a)
    print(i.english())
    print(i.by_country('Israel'))
    print(i.english2())

def test2():
    a = open("preql/simple2.pql").read()
    i = Interface()
    i(a)
    print(i.english_speakers())
    print(i.person_and_language())
    print(i.from_my_country())
    print(i.population_count())
    # print(i.citizens_list())  # TODO requires maintaining return type

test2()