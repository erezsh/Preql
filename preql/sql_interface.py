from .utils import benchmark

from .sql import Sql, CompiledSQL, Select, QueryBuilder, sqlite, postgres
from . import exceptions

from .pql_types import Primitive, null    # XXX Code smell?

class SqlInterface:
    def query(self, sql, subqueries=None, qargs=(), quiet=False, state=None):
        assert isinstance(sql, Sql), sql

        qb = QueryBuilder(self.target, parameters=state and [state.ns])

        if subqueries:
            subqs = [q.compile(qb).text for (name, q) in subqueries.items()]
            sql_code = 'WITH ' + ',\n     '.join(subqs) + '\n'
        else:
            sql_code = ''

        compiled = sql.compile(qb)
        sql_code += compiled.text
        c = self._conn.cursor()
        if self._debug and not quiet:
            print_sql(sql_code)

        try:
            c.execute(sql_code)
            # c.execute(sql_code, qargs)    # XXX messes up when sql_code contains '%', like for LIKE
        except Exception as e:
            # print_sql(sql_code)
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise exceptions.pql_DatabaseQueryError(None, msg%(sql_code, e))

        if sql.type is not null:
            res = c.fetchall()
            imp = sql.type.import_result
            return imp(res)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _old_addmany(self, table, cols, values):
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


def print_sql(sql):
    for i, s in enumerate(sql.split('\n')):
        print('/**/    ' if i else '/**/;;  ', s)


class PostgresInterface(SqlInterface):
    target = postgres

    def __init__(self, host, database, user, password, debug=True):
        import psycopg2
        self._conn = psycopg2.connect(host=host,database=database, user=user, password=password)
        self._debug = debug


class SqliteInterface(SqlInterface):
    target = sqlite

    def __init__(self, filename=None, debug=True):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._debug = debug
