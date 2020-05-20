from .sql import Sql, QueryBuilder, sqlite, postgres
from . import exceptions

from .pql_types import null


class SqlInterface:
    def __init__(self, debug=False):
        self._debug = debug

    def query(self, sql, subqueries=None, qargs=(), quiet=False, state=None):
        assert isinstance(sql, Sql), sql
        sql_code = self._compile_sql(sql, subqueries, qargs, state)

        if self._debug and not quiet:
            print_sql(sql_code)

        cur = self._execute_sql(sql_code)

        return self._import_result(sql.type, cur)

    def _import_result(self, sql_type, c):
        if sql_type is not null:
            res = c.fetchall()
            imp = sql_type.import_result
            return imp(res)

    def _execute_sql(self, sql_code):
        c = self._conn.cursor()

        try:
            c.execute(sql_code)
            # c.execute(sql_code, qargs)    # XXX messes up when sql_code contains '%', like for LIKE
        except Exception as e:
            # print_sql(sql_code)
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise exceptions.pql_DatabaseQueryError([], msg%(sql_code, e))

        return c


    def _compile_sql(self, sql, subqueries=None, qargs=(), state=None):
        qb = QueryBuilder(self.target, parameters=state and [state.ns])

        if subqueries:
            subqs = [q.compile(qb).text for (name, q) in subqueries.items()]
            sql_code = 'WITH ' + ',\n     '.join(subqs) + '\n'
        else:
            sql_code = ''
        compiled = sql.compile(qb)
        sql_code += compiled.text
        return sql_code


    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


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
