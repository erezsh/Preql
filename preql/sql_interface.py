from .utils import dataclass
from .loggers import sql_log

from .sql import Sql, QueryBuilder, sqlite, postgres
from . import exceptions

from .pql_types import T, Type, from_sql, Object


@dataclass
class Const(Object):
    type: Type
    value: object

class ConnectError(Exception):
    pass

class SqlInterface:
    def __init__(self, print_sql=False):
        self._print_sql = print_sql

    def query(self, sql, subqueries=None, qargs=(), quiet=False, state=None):
        assert state
        assert isinstance(sql, Sql), sql
        sql_code = self._compile_sql(sql, subqueries, qargs, state)

        if self._print_sql and not quiet:
            log_sql(sql_code)

        cur = self._execute_sql(sql_code)

        return self._import_result(sql.type, cur, state)

    def _import_result(self, sql_type, c, state):
        if sql_type is not T.null:
            res = c.fetchall()
            # imp = sql_type.import_result
            # return imp(res)
            return from_sql(state, Const(sql_type, res))

    def _execute_sql(self, sql_code):
        c = self._conn.cursor()

        try:
            c.execute(sql_code)
            # c.execute(sql_code, qargs)    # XXX messes up when sql_code contains '%', like for LIKE
        except Exception as e:
            # print_sql(sql_code)
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise exceptions.DatabaseQueryError(msg%(sql_code, e))

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

    def ping(self):
        c = self._conn.cursor()
        c.execute('select 1')
        row ,= c.fetchall()
        n ,= row
        assert n == 1

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def log_sql(sql):
    for i, s in enumerate(sql.split('\n')):
        prefix = '/**/    ' if i else '/**/;;  '
        sql_log.debug(prefix+s)


class PostgresInterface(SqlInterface):
    target = postgres

    def __init__(self, host, port, database, user, password, print_sql=True):
        import psycopg2
        try:
            self._conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        except psycopg2.OperationalError as e:
            raise ConnectError(*e.args) from e

        self._print_sql = print_sql


class SqliteInterface(SqlInterface):
    target = sqlite

    def __init__(self, filename=None, print_sql=True):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._print_sql = print_sql
