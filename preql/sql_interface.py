from decimal import Decimal
from .utils import classify, dataclass
from .loggers import sql_log

from .sql import Sql, QueryBuilder, sqlite, postgres, mysql, duck
from . import exceptions

from .pql_types import T, Type, Object, Id
from .types_impl import from_sql


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

        return self._execute_sql(sql.type, sql_code, state)
        # return self._import_result(sql.type, cur, state)

    def _import_result(self, sql_type, c, state):
        if sql_type is not T.nulltype:
            res = c.fetchall()
            # imp = sql_type.import_result
            # return imp(res)
            return from_sql(state, Const(sql_type, res))

    def _execute_sql(self, sql_type, sql_code, state):

        try:
            c = self._conn.cursor()
            c.execute(sql_code)
            # c.execute(sql_code, qargs)    # XXX messes up when sql_code contains '%', like for LIKE
        except Exception as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise exceptions.DatabaseQueryError(msg%(sql_code, e))

        return self._import_result(sql_type, c, state)


    def _compile_sql(self, sql, subqueries=None, qargs=(), state=None):
        qb = QueryBuilder(self.target)

        return sql.finalize_with_subqueries(qb, subqueries)

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

    def import_table_types(self, state):
        tables = self.list_tables()
        for table_name in tables:
            table_type = self.import_table_type(state, table_name)
            yield None, table_name, table_type



def log_sql(sql):
    for i, s in enumerate(sql.split('\n')):
        prefix = '/**/    ' if i else '/**/;;  '
        sql_log.debug(prefix+s)


class MysqlInterface(SqlInterface):
    target = mysql

    def __init__(self, host, port, database, user, password, print_sql=False):
        import mysql.connector
        from mysql.connector import errorcode

        args = dict(host=host, port=port, database=database, user=user, password=password)
        args = {k:v for k, v in args.items() if v is not None}

        try:
            # TODO utf8??
            self._conn = mysql.connector.connect(charset='utf8', use_unicode=True, **args)
        except mysql.connector.Error as e:
            if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise ConnectError("Bad user name or password") from e
            elif e.errno == errorcode.ER_BAD_DB_ERROR:
                raise ConnectError("Database does not exist") from e
            else:
                raise ConnectError(*e.args) from e

        self._print_sql = print_sql

    def table_exists(self, name):
        tables = [t.lower() for t in self.list_tables()]
        return name.lower() in tables

    def list_tables(self):
        sql_code = "SHOW TABLES"
        return self._execute_sql(T.list[T.string], sql_code, None)

    def import_table_type(self, state, name, columns_whitelist=None):
        columns_t = T.table(dict(
            name=T.string,
            type=T.string,
            nullable=T.string,
            key=T.string,
            default=T.string,
            extra=T.string,
        ))
        columns_q = "desc %s" % name
        sql_columns = self._execute_sql(columns_t, columns_q, state)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = {c['name']: _type_from_sql(c['type'].decode(), c['nullable']) for c in sql_columns}

        return T.table(cols, name=Id(name))


class PostgresInterface(SqlInterface):
    target = postgres

    def __init__(self, host, port, database, user, password, print_sql=False):
        import psycopg2
        import psycopg2.extras
        psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)
        try:
            self._conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        except psycopg2.OperationalError as e:
            raise ConnectError(*e.args) from e

        self._print_sql = print_sql


    def table_exists(self, name):
        sql_code = "SELECT count(*) FROM information_schema.tables where table_name='%s'" % name
        cnt = self._execute_sql(T.int, sql_code, None)
        return cnt > 0

    def list_tables(self):
        sql_code = "SELECT table_name FROM information_schema.tables where table_schema='public'"
        return self._execute_sql(T.list[T.string], sql_code, None)


    _schema_columns_t = T.table(dict(
        schema=T.string,
        table=T.string,
        name=T.string,
        pos=T.int,
        nullable=T.bool,
        type=T.string,
    ))

    def import_table_type(self, state, name, columns_whitelist=None):

        columns_q = """SELECT table_schema, table_name, column_name, ordinal_position, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_name = '%s'
            """ % name
        sql_columns = self._execute_sql(self._schema_columns_t, columns_q, state)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], _type_from_sql(c['type'], c['nullable'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=Id(name))

    def import_table_types(self, state):
        columns_q = """SELECT table_schema, table_name, column_name, ordinal_position, is_nullable, data_type
            FROM information_schema.columns
            """
        sql_columns = self._execute_sql(self._schema_columns_t, columns_q, state)

        columns_by_table = classify(sql_columns, lambda c: (c['schema'], c['table']))

        for (schema, table_name), columns in columns_by_table.items():
            cols = [(c['pos'], c['name'], _type_from_sql(c['type'], c['nullable'])) for c in columns]
            cols.sort()
            cols = dict(c[1:] for c in cols)

            # name = '%s.%s' % (schema, table_name)
            yield schema, table_name, T.table(cols, name=Id(schema, table_name))



class SqliteInterface(SqlInterface):
    target = sqlite

    def __init__(self, filename=None, print_sql=False):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._print_sql = print_sql

    def table_exists(self, name):
        sql_code = "SELECT count(*) FROM sqlite_master where name='%s' and type='table'" % name
        cnt = self._execute_sql(T.int, sql_code, None)
        return cnt > 0

    def list_tables(self):
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        return self._execute_sql(T.list[T.string], sql_code, None)


    table_schema_type = T.table(dict(
        pos=T.int,
        name=T.string,
        type=T.string,
        notnull=T.bool,
        default_value=T.string,
        pk=T.bool,
    ))

    def import_table_type(self, state, name, columns_whitelist=None):
        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q, state)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], _type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        pk = [[c['name']] for c in sql_columns if c['pk']]

        return T.table(cols, name=Id(name), pk=pk)

class DuckInterface(SqliteInterface):
    target = duck

    def __init__(self, filename=None, print_sql=False):
        import duckdb
        self._conn = duckdb.connect(filename or ':memory:')
        self._print_sql = print_sql

    def rollback(self):
        pass    # XXX


import subprocess
import json
class GitInterface(SqliteInterface):
    "Uses https://github.com/augmentable-dev/askgit"

    target = sqlite

    def __init__(self, path, print_sql):
        self.path = path
        self._print_sql = print_sql


    table_schema_type = T.table(dict(
        cid=T.int,
        dflt_value=T.string,
        name=T.string,
        notnull=T.bool,
        pk=T.bool,
        type=T.string,
    ))

    def _execute_sql(self, sql_type, sql_code, state):
        try:
            res = subprocess.check_output(['askgit', '--format', 'json', sql_code])
        except FileNotFoundError:
            msg = "Could not find executable 'askgit'. Make sure it's installed, and try again."
            raise exceptions.DatabaseQueryError(msg)
        except subprocess.CalledProcessError as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise exceptions.DatabaseQueryError(msg%(sql_code, e))

        return self._import_result(sql_type, res, state)

    def _import_result(self, sql_type, c, state):
        if sql_type is not T.nulltype:
            if sql_type <= T.table:
                lookup = dict(reversed(x) for x in enumerate(sql_type.elems))
                rows = [json.loads(x) for x in c.split(b'\n') if x.strip()]
                res = []
                for row in rows:
                    # TODO refactor into a function
                    x = ["PLACEHOLDER"] * len(row)
                    for k, v in row.items():
                        x[lookup[k]] = v
                    assert "PLACEHOLDER" not in x, (x, row)
                    res.append(x)
            else:
                res = [list(json.loads(x).values()) for x in c.split(b'\n') if x.strip()]

            return from_sql(state, Const(sql_type, res))

    def import_table_type(self, state, name, columns_whitelist=None):
        # TODO merge with superclass

        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q, state)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['cid'], c['name'], _type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=Id(name))

    def list_tables(self):
        # TODO merge with superclass?
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        res = self._execute_sql(T.table(dict(name=T.string)), sql_code, None)
        return [x['name'] for x in res]



def _bool_from_sql(n):
    if n == 'NO':
        n = False
    if n == 'YES':
        n = True
    assert isinstance(n, bool), n
    return n

def _type_from_sql(type, nullable):
    type = type.lower()
    d = {
        'integer': T.int,
        'int': T.int,           # mysql
        'tinyint(1)': T.bool,   # mysql
        'serial': T.t_id,
        'bigserial': T.t_id,
        'smallint': T.int,  # TODO smallint / bigint?
        'bigint': T.int,
        'character varying': T.string,
        'character': T.string,  # TODO char?
        'real': T.float,
        'float': T.float,
        'double precision': T.float,    # double on 32-bit?
        'boolean': T.bool,
        'timestamp': T.datetime,
        'timestamp without time zone': T.datetime,
        'text': T.text,
    }
    try:
        v = d[type]
    except KeyError:
        if type.startswith('int('): # TODO actually parse it
            return T.int
        elif type.startswith('tinyint('): # TODO actually parse it
            return T.int
        elif type.startswith('varchar('): # TODO actually parse it
            return T.string

        return T.string.as_nullable()

    nullable = _bool_from_sql(nullable)

    return v.replace(_nullable=nullable)