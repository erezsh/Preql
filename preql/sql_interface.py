import operator
from pathlib import Path
import subprocess
import json

import dsnparse

from .utils import classify, dataclass
from .loggers import sql_log
from .context import context

from .core.sql import Sql, QueryBuilder, sqlite, postgres, mysql, duck, bigquery, quote_id, snowflake, redshift
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

    supports_foreign_key = True
    requires_subquery_name = False
    id_type_decl = 'INTEGER'
    max_rows_per_query = 1024


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
        qb = QueryBuilder()

        return sql.finalize_with_subqueries(qb, subqueries)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def list_namespaces(self):
        return []

    def import_table_types(self):
        # Inefficient implementation but generic
        tables = self.list_tables()
        for table_name in tables:
            table_type = self.import_table_type(table_name)
            yield None, table_name, table_type

    def qualified_name(self, name):
        return name

    def quote_name(self, name):
        return f'"{name}"'



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

    id_type_decl = "INTEGER NOT NULL AUTO_INCREMENT"
    requires_subquery_name = True

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
        assert isinstance(name, Id)
        tables = [t.lower() for t in self.list_tables()]
        return name.lower() in tables

    def list_tables(self):
        sql_code = "SHOW TABLES"
        names = self._execute_sql(T.list[T.string], sql_code)
        return list(map(Id, names))

    def import_table_type(self, name, columns_whitelist=None):
        assert isinstance(name, Id)
        assert len(name.parts) == 1 # TODO !

        columns_t = T.table(dict(
            name=T.string,
            type=T.string,
            nullable=T.string,
            key=T.string,
            default=T.string,
            extra=T.string,
        ))
        columns_q = "desc %s" % name.name
        sql_columns = self._execute_sql(columns_t, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = {c['name']: type_from_sql(c['type'].decode(), c['nullable']) for c in sql_columns}

        return T.table(cols, name=name)

    def quote_name(self, name):
        return f'`{name}`'


class PostgresInterface(SqlInterfaceCursor):
    target = postgres

    id_type_decl = "SERIAL"
    requires_subquery_name = True

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




    def table_exists(self, table_id):
        assert isinstance(table_id, Id)
        if len(table_id.parts) == 1:
            schema = 'public'
            name ,= table_id.parts
        elif len(table_id.parts) == 2:
            schema, name = table_id.parts
        else:
            raise Signal.make(T.DbError, None, "Postgres doesn't support nested schemas")

        sql_code = f"SELECT count(*) FROM information_schema.tables where table_name='{name}' and table_schema='{schema}'"
        cnt = self._execute_sql(T.int, sql_code)
        return cnt > 0

    def list_tables(self):
        # TODO import more schemas?
        sql_code = "SELECT table_name FROM information_schema.tables where table_schema='public'"
        names = self._execute_sql(T.list[T.string], sql_code)
        return list(map(Id, names))


    _schema_columns_t = T.table(dict(
        schema=T.string,
        table=T.string,
        name=T.string,
        pos=T.int,
        nullable=T.bool,
        type=T.string,
    ))

    def import_table_type(self, table_id, columns_whitelist=None):
        assert isinstance(table_id, Id)

        if len(table_id.parts) == 1:
            schema = 'public'
            name ,= table_id.parts
        elif len(table_id.parts) == 2:
            schema, name = table_id.parts
        else:
            raise Signal.make(T.DbError, None, "Postgres doesn't support nested schemas")

        columns_q = f"""SELECT table_schema, table_name, column_name, ordinal_position, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_name = '{name}' AND table_schema = '{schema}'
            """
        sql_columns = self._execute_sql(self._schema_columns_t, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], type_from_sql(c['type'], c['nullable'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=table_id)

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


class RedshiftInterface(PostgresInterface):
    target = redshift

    id_type_decl = "INT IDENTITY(1,1)"


class SnowflakeInterface(SqlInterface):
    target = snowflake

    id_type_decl = "number autoincrement"
    max_rows_per_query = 16384

    def __init__(self, account, user, password, path, schema, database, print_sql=False):
        import logging
        logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
        import snowflake.connector

        self._client = snowflake.connector.connect(
            user=user,
            password=password,
            account=account
            )
        self._client.cursor().execute(f"USE WAREHOUSE {path.lstrip('/')}")
        self._client.cursor().execute(f"USE DATABASE {database}")
        self._client.cursor().execute(f"USE SCHEMA {schema}")

        self._print_sql = print_sql

    def quote_name(self, name):
        # TODO is this right?
        return name

    def table_exists(self, name):
        assert isinstance(name, Id)
        tables = [t.lower() for t in self.list_tables()]
        return name.lower() in tables

    def list_tables(self):
        sql_code = "SHOW TABLES"
        table_type = T.table(dict(
            created_on=T.datetime,
            name=T.string,
            database_name=T.string,
            schema_name=T.string,
            kind=T.string,
            comment=T.string,
            cluster_by=T.string,
            rows=T.int,
            bytes=T.int,
            owner=T.string,
            retention_time=T.string,
            dropped_on=T.string,
            automatic_clustering=T.string,
            change_tracking=T.string,
            search_optimization=T.string,
            search_optimization_progress=T.string,
            search_optimization_bytes=T.string,
            # is_exteral=T.string,
        ))
        
        tables = self._execute_sql(table_type, sql_code)
        return [Id(table['name']) for table in tables]

    def import_table_type(self, name, columns_whitelist=None):
        assert isinstance(name, Id)
        sql_code = f"DESC TABLE {quote_id(name)}"
        table_type = T.table(dict(
            name=T.string,
            type=T.string,
            kind=T.string,
            null=T.bool,
            default=T.string,
            primary_key=T.bool,
            unique_key=T.bool,
            check=T.string,
            expression=T.string,
            comment=T.string,
            policy=T.string,
        ))
        fields = self._execute_sql(table_type, sql_code)

        cols = {
            f['name']: type_from_sql(f['type'], f['null'] == 'Y')
            for f in fields
            if columns_whitelist is None or f.name in columns_whitelist
        }

        return T.table(cols, name=name)

    def _import_result(self, sql_type, c):
        if sql_type is T.nulltype:
            return None

        try:
            res = c.fetchall()
        except Exception as e:
            msg = "Exception when trying to fetch SQL result. Got error: %s"
            raise DatabaseQueryError(msg%(e))

        return sql_result_to_python(Const(sql_type, res))
    def _execute_sql(self, sql_type, sql_code):
        import snowflake.connector
        cs = self._client.cursor()
        try:
            res = cs.execute(sql_code)
            return self._import_result(sql_type, res)
        except snowflake.connector.errors.DatabaseError as e:
            msg = "Exception when trying to execute SQL code:\n    %s\n\nGot error: %s"
            raise DatabaseQueryError(msg%(sql_code, e))
        finally:
            cs.close()

class BigQueryInterface(SqlInterface):
    target = bigquery

    PREQL_DATASET = '_preql_d9e7334a9e028e8aa38509912dfc2aac'

    supports_foreign_key = False
    id_type_decl = 'STRING NOT NULL'

    def __init__(self, project, print_sql=False):
        from google.cloud import bigquery

        # job_config = bigquery.job.QueryJobConfig(default_dataset=f'{project}._preql')
        # self._client = bigquery.Client(project, default_query_job_config=job_config)
        self._client = bigquery.Client(project)
        self._default_dataset = None

        self._print_sql = print_sql

        self._dataset_ensured = False

    def quote_name(self, name):
        return f'`{name}`'

    def ensure_dataset(self):
        if self._dataset_ensured:
            return

        self._client.delete_dataset(self.PREQL_DATASET, delete_contents=True, not_found_ok=True)
        self._client.create_dataset(self.PREQL_DATASET)

        self._dataset_ensured = True



    def _list_tables(self):
        for ds in self._client.list_datasets():
            for t in self._client.list_tables(ds.reference):
                # yield t.full_table_id.replace(':', '.')     # Hacky
                yield Id(*t.full_table_id.replace(':', '.').split('.'))

    def list_tables(self):
        return list(self._list_tables())

    def list_namespaces(self):
        datasets = self._client.list_datasets()
        return [x.reference.dataset_id for x in datasets]

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


    def get_table(self, name):
        from google.api_core.exceptions import NotFound, BadRequest
        try:
            return self._client.get_table('.'.join(name.parts))
        except ValueError as e:
            raise Signal.make(T.ValueError, None, f'BigQuery error: {e}')
        except NotFound as e:
            raise Signal.make(T.DbQueryError, None, str(e))
        except BadRequest as e:
            raise Signal.make(T.DbQueryError, None, str(e))


    def table_exists(self, name):
        try:
            self.get_table(name)
        except Signal:
            return False
        return True


    def import_table_type(self, name, columns_whitelist=None):
        assert isinstance(name, Id)

        schema = self.get_table(name).schema

        cols = {
            f.name: type_from_sql(f.field_type, f.is_nullable)
            for f in schema
            if columns_whitelist is None or f.name in columns_whitelist
        }

        return T.table(cols, name=name)

    def import_table_types(self):
        # Inefficient implementation
        tables = self.list_tables()
        for table_id in tables:
            table_type = self.import_table_type(table_id)

            # XXX support nested schemas
            schema = table_id.parts[:-1]
            name = table_id.parts[-1]
            yield schema[-1], name, table_type

    def list_datasets(self):
        return [ds.dataset_id for ds in self._client.list_datasets()]

    def set_default_dataset(self, dataset):
        self._default_dataset = dataset

    def get_default_dataset(self):
        # if self._default_dataset is None:
        #     datasets = self.list_datasets()
        #     if not datasets:
        #         raise Signal.make(T.ValueError, None, "No dataset found.")
        #     self._default_dataset = datasets[0]
        # return self._default_dataset
        self.ensure_dataset()
        return self.PREQL_DATASET

    def qualified_name(self, name):
        "Ensure the name has a dataset"
        assert isinstance(name, Id)
        if len(name.parts) > 1:
            # already has dataset
            return name
        return Id(self.get_default_dataset(), name.parts[-1])

    def rollback(self):
        # XXX No error? No warning?
        pass
    def commit(self):
        # XXX No error? No warning?
        pass

    def close(self):
        self._client.close()

class AbsSqliteInterface:
    def table_exists(self, name):
        assert isinstance(name, Id), name
        if len(name.parts) > 1:
            raise Signal.make(T.DbError, None, "Sqlite does not implement namespaces")

        sql_code = "SELECT count(*) FROM sqlite_master where name='%s' and type='table'" % name.name
        cnt = self._execute_sql(T.int, sql_code)
        return cnt > 0

    def list_tables(self):
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        return [Id(x) for x in self._execute_sql(T.list[T.string], sql_code)]

    table_schema_type = T.table(dict(
        pos=T.int,
        name=T.string,
        type=T.string,
        notnull=T.bool,
        default_value=T.string,
        pk=T.bool,
    ))

    def import_table_type(self, name, columns_whitelist=None):
        assert isinstance(name, Id), name

        columns_q = """pragma table_info(%s)""" % quote_id(name)
        sql_columns = self._execute_sql(self.table_schema_type, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['pos'], c['name'], type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        pk = [[c['name']] for c in sql_columns if c['pk']]

        return T.table(cols, name=name, pk=pk)


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

    def quote_name(self, name):
        return f'[{name}]'


class DuckInterface(SqliteInterface):
    target = duck

    supports_foreign_key = False
    requires_subquery_name = True

    def _create_connection(self):
        import duckdb
        return duckdb.connect(self._filename or ':memory:')

    def quote_name(self, name):
        return f'"{name}"'

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
        assert isinstance(name, Id)
        assert len(name.parts) == 1 # TODO !
        # TODO merge with superclass

        columns_q = """pragma table_info('%s')""" % name
        sql_columns = self._execute_sql(self.table_schema_type, columns_q)

        if columns_whitelist:
            wl = set(columns_whitelist)
            sql_columns = [c for c in sql_columns if c['name'] in wl]

        cols = [(c['cid'], c['name'], type_from_sql(c['type'], not c['notnull'])) for c in sql_columns]
        cols.sort()
        cols = dict(c[1:] for c in cols)

        return T.table(cols, name=name)

    def list_tables(self):
        # TODO merge with superclass?
        sql_code = "SELECT name FROM sqlite_master where type='table'"
        res = self._execute_sql(T.table(dict(name=T.string)), sql_code)
        return [Id(x['name']) for x in res]


def _drop_tables(state, *tables):
    # XXX temporary. Used for testing
    db = state.db
    with context(state=state):
        for t in tables:
            t = quote_id(db.qualified_name(t))
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

    if len(dsn.paths) == 0:
        path = ''
    elif len(dsn.paths) == 1:
        path ,= dsn.paths
    else:
        raise ValueError("Bad value for uri, too many paths: %s" % db_uri)

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
    elif scheme == 'snowflake':
        return SnowflakeInterface(dsn.host, dsn.user, dsn.password, path, **dsn.query, print_sql=print_sql)
    if scheme == 'redshift':
        return RedshiftInterface(dsn.host, dsn.port, path, dsn.user, dsn.password, print_sql=print_sql)

    raise NotImplementedError(f"Scheme {dsn.scheme} currently not supported")
