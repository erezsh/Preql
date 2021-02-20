from contextlib import contextmanager


from . import settings
from . import pql_ast as ast
from . import pql_objects as objects
from .utils import classify
from .interpreter import Interpreter
from .evaluate import cast_to_python, localize, evaluate
from .interp_common import create_engine, call_pql_func
from .pql_types import T
from .pql_functions import import_pandas
from .context import context
from . import sql

from . import display
display.install_reprs()


def _call_pql_func(state, name, args):
    with context(state=state):
        count = call_pql_func(state, name, args)
        return cast_to_python(state, count)


class TablePromise:
    """Returned by Preql whenever the result is a table

    Fetching values creates queries to database engine
    """

    def __init__(self, state, inst):
        self._state = state
        self._inst = inst
        self._rows = None

    def to_json(self):
        "Returns table as a list of rows, i.e. ``[{col1: value, col2: value, ...}, ...]``"
        if self._rows is None:
            self._rows = cast_to_python(self._state, self._inst)
        assert self._rows is not None
        return self._rows

    def to_pandas(self):
        "Returns table as a Pandas dataframe (requires pandas installed)"
        from pandas import DataFrame
        return DataFrame(self)

    def __eq__(self, other):
        """Compare the table to a JSON representation of it as list of objects

        Essentially: ``return self.to_json() == other``
        """
        return self.to_json() == other

    def __len__(self):
        "Run a count query on table"
        return _call_pql_func(self._state, 'count', [self._inst])

    def __iter__(self):
        return iter(self.to_json())

    def __getitem__(self, index):
        "Run a slice query on table"
        with context(state=self._state):
            if isinstance(index, slice):
                offset = index.start or 0
                limit = index.stop - offset
                return call_pql_func(self._state, 'limit_offset', [self._inst, ast.make_const(limit), ast.make_const(offset)])

            # TODO different debug log level / mode
            res ,= cast_to_python(self._state, self[index:index+1])
            return res

    def __repr__(self):
        return repr(self.to_json())


def promise(state, inst):
    if inst.type <= T.table:
        return TablePromise(state, inst)

    return localize(state, inst)


class Preql:
    """Provides an API to run Preql code from Python

    Example:
        >>> import preql
        >>> p = preql.Preql()
        >>> p('[1, 2]{item+1}')
        [2, 3]
    """

    __name__ = "Preql"

    def __init__(self, db_uri: str='sqlite://:memory:', print_sql: bool=settings.print_sql, auto_create: bool = False):
        """Initialize a new Preql instance

        Parameters:
            db_uri (str, optional): URI of database. Defaults to using a non-persistent memory database.
            print_sql (bool, optional): Whether or not to print every SQL query that is executed (default defined in settings)
        """
        self._db_uri = db_uri
        self._print_sql = print_sql
        # self.engine.ping()

        engine = create_engine(self._db_uri, print_sql=self._print_sql, auto_create=auto_create)
        self._reset_interpreter(engine)

    def set_output_format(self, fmt):
        self.interp.state.fmt = fmt  # TODO proper api

    def _reset_interpreter(self, engine=None):
        if engine is None:
            engine = self.interp.state.db
        self.interp = Interpreter(engine)
        self.interp.state._py_api = self # TODO proper api

    def close(self):
        self.interp.state.db.close()

    def __getattr__(self, fname):
        var = self.interp.state.get_var(fname)
        if isinstance(var, objects.Function):
            def delegate(*args, **kw):
                assert not kw
                pql_args = [objects.from_python(a) for a in args]
                pql_res = self.interp.call_func(fname, pql_args)
                return self._wrap_result( pql_res )
            return delegate
        else:
            return self._wrap_result( evaluate( self.interp.state, var ))

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        assert not isinstance(res, ast.Ast), res
        return promise(self.interp.state, res)  # TODO session, not state

    def _run_code(self, pq: str, source_file: str, **args):
        pql_args = {name: objects.from_python(value) for name, value in args.items()}
        return self.interp.execute_code(pq + "\n", source_file, pql_args)

    def __call__(self, pq, **args):
        res = self._run_code(pq, '<inline>', **args)
        if res:
            return self._wrap_result(res)

    def load(self, filename, rel_to=None):
        """Load a Preql script

        Parameters:
            filename (str): Name of script to run
            rel_to (Optional[str]): Path to which ``filename`` is relative.
        """
        self.interp.include(filename, rel_to)

    @contextmanager
    def transaction(self):
        try:
            yield self  # TODO new instance?
        except:
            self.rollback()
            raise

        self.commit()

    def start_repl(self, *args):
        "Run the interactive prompt"
        from .repl import start_repl
        start_repl(self, *args)

    def commit(self):
        return self.interp.state.db.commit()

    def rollback(self):
        return self.interp.state.db.rollback()

    def _drop_tables(self, *tables):
        state = self.interp.state
        # XXX temporary. Used for testing
        for t in tables:
            t = sql._quote(state.db.target, state.db.qualified_name(t))
            state.db._execute_sql(T.nulltype, f"DROP TABLE {t};", state)

    def import_pandas(self, **dfs):
        """Import pandas.DataFrame instances into SQL tables

        Example:
            >>> pql.import_pandas(a=df_a, b=df_b)
        """
        with self.interp.setup_context():
            return list(import_pandas(self.interp.state, dfs))


    def load_all_tables(self):
        table_types = self.interp.state.db.import_table_types(self.interp.state)
        table_types_by_schema = classify(table_types, lambda x: x[0], lambda x: x[1:])

        for schema_name, table_types in table_types_by_schema.items():
            if schema_name:
                schema = objects.Module(schema_name, {})
                self.interp.set_var(schema_name, schema)

            for table_name, table_type in table_types:
                db_name = table_type.options['name']
                inst = objects.new_table(table_type, db_name)

                if schema_name:
                    schema.namespace[table_name] = inst
                else:
                    if not self.interp.has_var(table_name):
                        self.interp.set_var(table_name, inst)




#     def _functions(self):
#         return {name:f for name,f in self.interp.state.namespace.items()
#                 if isinstance(f, ast.FunctionDef)}

#     def add_many(self, table, values):
#         cols = [c.name
#                 for c in self.interp.state.namespace[table].columns.values()
#                 if not isinstance(c.type, (ast.BackRefType, ast.IdType))]
#         return self.engine.addmany(table, cols, values)

#     def add(self, table, values):
#         return self.add_many(table, [values])
