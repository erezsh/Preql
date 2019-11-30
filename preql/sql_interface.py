from .sql import Sql, CompiledSQL

class SqlInterface:
    pass

def print_sql(sql):
    for i, s in enumerate(sql.split('\n')):
        print('#-  ' if i else '#?  ', s)


class SqliteInterface(SqlInterface):
    def __init__(self, filename=None, debug=True):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._debug = debug

    def query(self, sql, subqueries=(), qargs=(), quiet=False):
        assert isinstance(sql, Sql), sql
        if subqueries:
            subqs = [f"{name} AS ({q.compile().text})" for (name, q) in subqueries.items()]
            sql_code = 'WITH ' + ',\n     '.join(subqs) + '\nSELECT * FROM '
        else:
            sql_code = ''
        compiled = sql.compile()
        sql_code += compiled.text
        c = self._conn.cursor()
        if self._debug and not quiet:
            print_sql(sql_code)
        c.execute(sql_code, qargs)
        res = c.fetchall()
        try:
            imp = compiled.sql.type.import_result
        except AttributeError:
            return res

        return imp(res)

    def commit(self):
        self._conn.commit()

    def addmany(self, table, cols, values):
        assert all(len(v)==len(cols) for v in values), (cols, values[0])

        c = self._conn.cursor()
        qmarks = ','.join(['?'] * len(cols))
        cols_str = ','.join(cols)
        sql = f'INSERT INTO {table} ({cols_str}) VALUES ({qmarks})'
        if self._debug:
            print_sql(sql)
        ids = []
        for v in values:
            c.execute(sql, v)
            ids.append(c.lastrowid)
        # c.executemany(sql, values)
        assert len(ids) == len(set(ids))
        inserted = len(ids)
        if self._debug:
            print('#-- Inserted %d rows' % inserted)
        assert inserted == len(values)
        return ids

