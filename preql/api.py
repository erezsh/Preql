from .compiler import Compiler


class DbEngine:
    def __call__(self, q, *args, **kw):
        return self._execute(q, *args, **kw)

    def __getattr__(self, fname):
        def delegate(*args, **kw):
            f, params = self._compiler.compile_func_call(fname)
            return self._query(f, params , *args, **kw)
        return delegate

    def __getitem__(self, pq):
        sql = self._compiler.compile_query(pq)
        return self._query(sql, [])

class SqliteEngine(DbEngine):
    def __init__(self, filename=None):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._compiler = Compiler()

    def _query(self, sql, qargs, *args, **kw):
        assert len(args) == len(qargs)
        dargs = dict(zip(qargs, args))
        c = self._conn.cursor()
        c.execute(sql, dargs)
        return c.fetchall()

    def _execute(self, pqcode):
        c = self._conn.cursor()

        for sq in self._compiler.compile_statements(pqcode):
            print('##', repr(sq))
            c.execute(sq)
            assert not c.fetchall()


def test():
    a = open("preql/simple1.pql").read()
    e = SqliteEngine()
    e._execute(a)
    print(e.english())

test()