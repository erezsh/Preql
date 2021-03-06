import operator
from pathlib import Path
import subprocess
import json

import dsnparse

from .utils import classify, dataclass
from .loggers import sql_log
from .context import context

from .core.sql import Sql, QueryBuilder, sqlite, postgres, mysql, duck, from_sql, bigquery, _quote
from .core.pql_types import T, Type, Object, Id
from .core.exceptions import DatabaseQueryError, Signal

@dataclass
class Const(Object):
    type: Type
    value: object

class ConnectError(Exception):
    pass


def log_sql(sql):
    for i, s in enumerate(sql.split('\n')):
        prefix = '/**/    ' if i else '/**/;;  '
        sql_log.debug(prefix+s)


class SqlInterface:
    _conn: object

    def __init__(self, print_sql=False):
        self._print_sql = print_sql

    def query(self, sql, subqueries=None, qargs=(), quiet=False):
        assert context.state
        assert isinstance(sql, Sql), sql
        sql_code = self.compile_sql(sql, subqueries, qargs)

        if self._print_sql and not quiet:
            log_sql(sql_code)

        return self._execute_sql(sql.type, sql_code)
        # return self._import_result(sql.type, cur, state)

    def _import_result(self, sql_type, c):
        if sql_type is not T.nulltype:
            try:
                res = c.fetchall()
            except Exception as e:
                msg = "Exception when trying to fetch SQL result. Got error: %s"
                raise DatabaseQueryError(msg%(e))

            return from_sql(Const(sql_type, res))



    def compile_sql(self, sql, subqueries=None, qargs=()):
        qb = QueryBuilder(self.target)

        return sql.finalize_with_subqueries(qb, subqueries)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def import_table_types(self):
        tables = self.list_tables()
        for table_name in tables:
            table_type = self.import_table_type(table_name)
            yield None, table_name, table_type

    def qualified_name(self, name):
        return name


class SqlInterfaceCursor(SqlInterface):
    "An interface that uses the standard SQL cursor interface"

    def _backend_execute_sql(self, sql_code):
        c = self._conn.cursor()
        c.execute(sql_code)
        return c

    def _execute_sql(self, sql_type, sql_code):
        assert context.state
        try:
            c = self._backend_execute_sql(sql_code)
        except Exception as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise DatabaseQueryError(msg%(sql_code, e))

        return self._import_result(sql_type, c)

    def ping(self):
        c = self._conn.cursor()
        c.execute('select 1')
        row ,= c.fetchall()
        n ,= row
        assert n == 1


class MysqlInterface(SqlInterfaceCursor):
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
        return self._execute_sql(T.list[T.string], sql_code)

    def import_table_type(self, name, columns_whitelist=None):
        columns_t = T.table(dict(
            name=T.string,
            type=T.string,
            nullable=T.string,
            key=T.string,
            default=T.string,
            extra=T.string,
        ))
        columns_q = "desc %s" % name
        sql_columns = self._execute_sql(columns_t, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = {c['name']: _type_from_sql(c['type'].decode(), c['nullable']) for c in sql_columns}

        return T.table(cols, name=Id(name))


class PostgresInterface(SqlInterfaceCursor):
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
        cnt = self._execute_sql(T.int, sql_code)
        return cnt > 0

    def list_tables(self):
        sql_code = "SELECT table_name FROM information_schema.tables where table_schema='public'"
        return self._execute_sql(T.list[T.string], sql_code)


    _schema_columns_t = T.table(dict(
        schema=T.string,
        table=T.string,
        name=T.string,
        pos=T.int,
        nullable=T.bool,
        type=T.string,
    ))

    def import_table_type(self, name, columns_whitelist=None):

        columns_q = """SELECT table_schema, table_name, column_name, ordinal_position, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_name = '%s'
            """ % name
        sql_columns = self._execute_sql(self._schema_columns_t, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], _type_from_sql(c['type'], c['nullable'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=Id(name))

    def import_table_types(self):
        columns_q = """SELECT table_schema, table_name, column_name, ordinal_position, is_nullable, data_type
            FROM information_schema.columns
            """
        sql_columns = self._execute_sql(self._schema_columns_t, columns_q)

        columns_by_table = classify(sql_columns, lambda c: (c['schema'], c['table']))

        for (schema, table_name), columns in columns_by_table.items():
            cols = [(c['pos'], c['name'], _type_from_sql(c['type'], c['nullable'])) for c in columns]
            cols.sort()
            cols = dict(c[1:] for c in cols)

            # name = '%s.%s' % (schema, table_name)
            yield schema, table_name, T.table(cols, name=Id(schema, table_name))



class BigQueryInterface(SqlInterface):
    target = bigquery

    def __init__(self, project, print_sql=False):
        from google.cloud import bigquery

        self._client = bigquery.Client(project)
        self._default_dataset = None

        self._print_sql = print_sql


    def table_exists(self, name):
        from google.api_core.exceptions import NotFound
        try:
            self._client.get_table(name)
        except NotFound:
            return False
        return True

    def _list_tables(self):
        for ds in self._client.list_datasets():
            for t in self._client.list_tables(ds.reference):
                yield t.full_table_id.replace(':', '.')     # Hacky

    def list_tables(self):
        return list(self._list_tables())


    def _execute_sql(self, sql_type, sql_code):
        assert context.state
        try:
            res = list(self._client.query(sql_code))
        except Exception as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise DatabaseQueryError(msg%(sql_code, e))

        if sql_type is not T.nulltype:
            res = [list(i.values()) for i in res]
            return from_sql(Const(sql_type, res))



    _schema_columns_t = T.table(dict(
        schema=T.string,
        table=T.string,
        name=T.string,
        pos=T.int,
        nullable=T.bool,
        type=T.string,
    ))

    def import_table_type(self, name, columns_whitelist=None):
        cols = {}
        for f in self._client.get_table(name).schema:
            if columns_whitelist is None or f.name in columns_whitelist:
                cols[f.name] = _type_from_sql(f.field_type, f.is_nullable)

        return T.table(cols, name=Id(name))

    def list_datasets(self):
        return [ds.dataset_id for ds in self._client.list_datasets()]

    def set_default_dataset(self, dataset):
        self._default_dataset = dataset

    def get_default_dataset(self):
        if self._default_dataset is None:
            datasets = self.list_datasets()
            if not datasets:
                raise Signal(T.ValueError, None, "No dataset found.")
            self._default_dataset = datasets[0]
        return self._default_dataset

    def qualified_name(self, name):
        if '.' in name: # already has dataset
            return name
        return self.get_default_dataset() + '.' + name

    def rollback(self):
        # XXX No error? No warning?
        pass
    def commit(self):
        # XXX No error? No warning?
        pass

class AbsSqliteInterface:
    def table_exists(self, name):
        sql_code = "SELECT count(*) FROM sqlite_master where name='%s' and type='table'" % name
        cnt = self._execute_sql(T.int, sql_code)
        return cnt > 0

    def list_tables(self):
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        return self._execute_sql(T.list[T.string], sql_code)

    table_schema_type = T.table(dict(
        pos=T.int,
        name=T.string,
        type=T.string,
        notnull=T.bool,
        default_value=T.string,
        pk=T.bool,
    ))

    def import_table_type(self, name, columns_whitelist=None):
        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], _type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        pk = [[c['name']] for c in sql_columns if c['pk']]

        return T.table(cols, name=Id(name), pk=pk)


class _SqliteProduct:
    def __init__(self):
        self.product = 1

    def step(self, value):
        self.product *= value

    def finalize(self):
        return self.product

class SqliteInterface(SqlInterfaceCursor, AbsSqliteInterface):
    target = sqlite

    def __init__(self, filename=None, print_sql=False):
        import sqlite3
        # sqlite3.enable_callback_tracebacks(True)
        try:
            self._conn = sqlite3.connect(filename or ':memory:')
        except sqlite3.OperationalError as e:
            raise ConnectError(*e.args) from e

        def sqlite_throw(x):
            raise Exception(x)
        self._conn.create_function("power", 2, operator.pow)
        self._conn.create_function("_pql_throw", 1, sqlite_throw)
        self._conn.create_aggregate("_pql_product", 1, _SqliteProduct)

        self._print_sql = print_sql



class DuckInterface(AbsSqliteInterface):
    target = duck

    def __init__(self, filename=None, print_sql=False):
        import duckdb
        self._conn = duckdb.connect(filename or ':memory:')
        self._print_sql = print_sql

    def rollback(self):
        pass    # XXX


class GitInterface(AbsSqliteInterface):
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

    def _execute_sql(self, sql_type, sql_code):
        assert context.state
        try:
            res = subprocess.check_output(['askgit', '--format', 'json', sql_code])
        except FileNotFoundError:
            msg = "Could not find executable 'askgit'. Make sure it's installed, and try again."
            raise DatabaseQueryError(msg)
        except subprocess.CalledProcessError as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise DatabaseQueryError(msg%(sql_code, e))

        return self._import_result(sql_type, res)

    def _import_result(self, sql_type, c):
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

            return from_sql(Const(sql_type, res))

    def import_table_type(self, name, columns_whitelist=None):
        # TODO merge with superclass

        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q)

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
        res = self._execute_sql(T.table(dict(name=T.string)), sql_code)
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


def _drop_tables(state, *tables):
    # XXX temporary. Used for testing
    db = state.db
    with context(state=state):
        for t in tables:
            t = _quote(db.target, db.qualified_name(t))
            db._execute_sql(T.nulltype, f"DROP TABLE {t};")


def create_engine(db_uri, print_sql, auto_create):
    dsn = dsnparse.parse(db_uri)
    if len(dsn.paths) != 1:
        raise ValueError("Bad value for uri: %s" % db_uri)
    path ,= dsn.paths
    if len(dsn.schemes) > 1:
        raise NotImplementedError("Preql doesn't support multiple schemes")
    scheme ,= dsn.schemes
    if scheme == 'sqlite':
        if not auto_create and path != ':memory:':
            if not Path(path).exists():
                raise ConnectError("File %r doesn't exist. To create it, set auto_create to True" % path)
        return SqliteInterface(path, print_sql=print_sql)
    elif scheme == 'postgres':
        return PostgresInterface(dsn.host, dsn.port, path, dsn.user, dsn.password, print_sql=print_sql)
    elif scheme == 'mysql':
        return MysqlInterface(dsn.host, dsn.port, path, dsn.user, dsn.password, print_sql=print_sql)
    elif scheme == 'git':
        return GitInterface(path, print_sql=print_sql)
    elif scheme == 'duck':
        return DuckInterface(path, print_sql=print_sql)
    elif scheme == 'bigquery':
        return BigQueryInterface(path, print_sql=print_sql)

    raise NotImplementedError(f"Scheme {dsn.scheme} currently not supported")
