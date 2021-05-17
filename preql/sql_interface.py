import operator
from pathlib import Path
import subprocess
import json

import dsnparse

from .utils import classify, dataclass
from .loggers import sql_log
from .context import context

from .core.sql import Sql, QueryBuilder, sqlite, postgres, mysql, duck, bigquery, _quote
from .core.sql_import_result import sql_result_to_python, type_from_sql
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
        # Inefficient implementation but generic
        tables = self.list_tables()
        for table_name in tables:
            table_type = self.import_table_type(table_name)
            yield None, table_name, table_type

    def qualified_name(self, name):
        return name


# from multiprocessing import Queue
import queue
import threading
from time import sleep

class TaskQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._task_results = {}
        self._closed = False
        self._start_worker()

    def _add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        task_id = object()
        self._queue.put((task_id, task, args, kwargs))
        return task_id

    def _start_worker(self):
        self.worker = t = threading.Thread(target=self._worker)
        t.daemon = True
        t.start()

    def _worker(self):
        while not self._closed:
            try:
                task_id, item, args, kwargs = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                res = item(*args, **kwargs)  
            except Exception as e:
                res = e
            self._task_results[task_id] = res
            self._queue.task_done()


    def _get_result(self, task_id):
        while task_id not in self._task_results:
            sleep(0.001)
        res = self._task_results.pop(task_id)
        if isinstance(res, Exception):
            raise res
        return res

    def run_task(self, task, *args, **kwargs):
        task_id = self._add_task(task, *args, **kwargs)
        return self._get_result(task_id)

    def close(self):
        self._closed = True



class BaseConnection:
    pass

class ThreadedConnection(BaseConnection):
    def __init__(self, create_connection):
        self._queue = TaskQueue()
        self._conn = self._queue.run_task(create_connection)

    def _backend_execute_sql(self, sql_code):
        c = self._conn.cursor()
        c.execute(sql_code)
        return c

    def _import_result(self, sql_type, c):
        if sql_type is T.nulltype:
            return None

        try:
            res = c.fetchall()
        except Exception as e:
            msg = "Exception when trying to fetch SQL result. Got error: %s"
            raise DatabaseQueryError(msg%(e))

        return sql_result_to_python(Const(sql_type, res))

    def _execute_sql(self, state, sql_type, sql_code):
        assert state
        with context(state=state):
            try:
                c = self._backend_execute_sql(sql_code)
            except Exception as e:
                msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
                raise DatabaseQueryError(msg%(sql_code, e))

            return self._import_result(sql_type, c)

    def execute_sql(self, sql_type, sql_code):
        return self._queue.run_task(self._execute_sql, context.state, sql_type, sql_code)


    def commit(self):
        self._queue.run_task(self._conn.commit)

    def rollback(self):
        self._queue.run_task(self._conn.rollback)

    def close(self):
        self._queue.run_task(self._conn.close)
        self._queue.close()




class SqlInterfaceCursor(SqlInterface):
    "An interface that uses the standard SQL cursor interface"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self._conn = ThreadedConnection(self._create_connection)


    def _execute_sql(self, sql_type, sql_code):
        return self._conn.execute_sql(sql_type, sql_code)

    def ping(self):
        c = self._conn.cursor()
        c.execute('select 1')
        row ,= c.fetchall()
        n ,= row
        assert n == 1



class MysqlInterface(SqlInterfaceCursor):
    target = mysql

    def __init__(self, host, port, database, user, password, print_sql=False):
        self._print_sql = print_sql

        args = dict(host=host, port=port, database=database, user=user, password=password)
        self._args = {k:v for k, v in args.items() if v is not None}
        super().__init__(print_sql)

    def _create_connection(self):
        import mysql.connector
        from mysql.connector import errorcode

        try:
            return mysql.connector.connect(charset='utf8', use_unicode=True, **self._args)
        except mysql.connector.Error as e:
            if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise ConnectError("Bad user name or password") from e
            elif e.errno == errorcode.ER_BAD_DB_ERROR:
                raise ConnectError("Database does not exist") from e
            else:
                raise ConnectError(*e.args) from e

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

        cols = {c['name']: type_from_sql(c['type'].decode(), c['nullable']) for c in sql_columns}

        return T.table(cols, name=Id(name))


class PostgresInterface(SqlInterfaceCursor):
    target = postgres

    def __init__(self, host, port, database, user, password, print_sql=False):
        self.args = dict(host=host, port=port, database=database, user=user, password=password)
        super().__init__(print_sql)

    def _create_connection(self):
        import psycopg2
        import psycopg2.extras
        psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)
        try:
            return psycopg2.connect(**self.args)
        except psycopg2.OperationalError as e:
            raise ConnectError(*e.args) from e




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

        cols = [(c['pos'], c['name'], type_from_sql(c['type'], c['nullable'])) for c in sql_columns]
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
            cols = [(c['pos'], c['name'], type_from_sql(c['type'], c['nullable'])) for c in columns]
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
            return sql_result_to_python(Const(sql_type, res))



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
                cols[f.name] = type_from_sql(f.field_type, f.is_nullable)

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

        cols = [(c['pos'], c['name'], type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
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

import math
class _SqliteStddev:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))



class SqliteInterface(SqlInterfaceCursor, AbsSqliteInterface):
    target = sqlite

    def __init__(self, filename=None, print_sql=False):
        self._filename = filename
        super().__init__(print_sql)


    def _create_connection(self):
        import sqlite3
        # sqlite3.enable_callback_tracebacks(True)
        try:
            conn = sqlite3.connect(self._filename or ':memory:')
        except sqlite3.OperationalError as e:
            raise ConnectError(*e.args) from e

        def sqlite_throw(x):
            raise Exception(x)

        conn.create_function("power", 2, operator.pow)
        conn.create_function("_pql_throw", 1, sqlite_throw)
        conn.create_aggregate("_pql_product", 1, _SqliteProduct)
        conn.create_aggregate("stddev", 1, _SqliteStddev)
        return conn


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

            return sql_result_to_python(Const(sql_type, res))

    def import_table_type(self, name, columns_whitelist=None):
        # TODO merge with superclass

        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['cid'], c['name'], type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=Id(name))

    def list_tables(self):
        # TODO merge with superclass?
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        res = self._execute_sql(T.table(dict(name=T.string)), sql_code)
        return [x['name'] for x in res]




def _drop_tables(state, *tables):
    # XXX temporary. Used for testing
    db = state.db
    with context(state=state):
        for t in tables:
            t = _quote(db.target, db.qualified_name(t))
            db._execute_sql(T.nulltype, f"DROP TABLE {t};")


_SQLITE_SCHEME = 'sqlite://'

def create_engine(db_uri, print_sql, auto_create):
    if db_uri.startswith(_SQLITE_SCHEME):
        # Parse sqlite:// ourselves, to allow for sqlite://c:/path/to/db
        path = db_uri[len(_SQLITE_SCHEME):]
        if not auto_create and path != ':memory:':
            if not Path(path).exists():
                raise ConnectError("File %r doesn't exist. To create it, set auto_create to True" % path)
        return SqliteInterface(path, print_sql=print_sql)

    dsn = dsnparse.parse(db_uri)
    if len(dsn.schemes) > 1:
        raise NotImplementedError("Preql doesn't support multiple schemes")
    scheme ,= dsn.schemes

    if len(dsn.paths) != 1:
        raise ValueError("Bad value for uri: %s" % db_uri)
    path ,= dsn.paths

    if scheme == 'postgres':
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
