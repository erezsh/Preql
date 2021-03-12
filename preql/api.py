from contextlib import contextmanager

from . import settings
from .core.pql_ast import pyvalue, Ast
from .core import pql_objects as objects
from .core.interpreter import Interpreter
from .core.pql_types import T
from .sql_interface import create_engine

from .core import display
display.install_reprs()


class TablePromise:
    """Returned by Preql whenever the result is a table

    Fetching values creates queries to database engine
    """

    def __init__(self, interp, inst):
        self._interp = interp
        self._inst = inst
        self._rows = None

    def to_json(self):
        "Returns table as a list of rows, i.e. ``[{col1: value, col2: value, ...}, ...]``"
        if self._rows is None:
            self._rows = self._interp.cast_to_python(self._inst)
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
        count = self._interp.call_builtin_func('count', [self._inst])
        return self._interp.cast_to_python(count)

    def __iter__(self):
        return iter(self.to_json())

    def __getitem__(self, index):
        "Run a slice query on table"
        if isinstance(index, slice):
            offset = index.start or 0
            limit = index.stop - offset
            return self._interp.call_builtin_func('limit_offset', [self._inst, pyvalue(limit), pyvalue(offset)])

        # TODO different debug log level / mode
        res ,= self._interp.cast_to_python(self[index:index+1])
        return res

    def __repr__(self):
        return repr(self.to_json())


def _prepare_instance_for_user(interp, inst):
    if inst.type <= T.table:
        return TablePromise(interp, inst)

    return interp.localize_obj(inst)


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
        self._auto_create = auto_create
        self._display = display.RichDisplay()
        # self.engine.ping()

        engine = create_engine(self._db_uri, print_sql=self._print_sql, auto_create=auto_create)
        self._reset_interpreter(engine)

    def set_output_format(self, fmt):
        if fmt == 'html':
            self._display = display.HtmlDisplay()
        else:
            self._display = display.RichDisplay()

        self.interp.state.display = self._display  # TODO proper api


    def _reset_interpreter(self, engine=None):
        if engine is None:
            engine = self.interp.state.db
        self.interp = Interpreter(engine, self._display)
        self.interp.state._py_api = self # TODO proper api

    def close(self):
        self.interp.state.db.close()

    def __getattr__(self, fname):
        var = self.interp.state.get_var(fname)

        if isinstance(var, objects.Function):
            def delegate(*args, **kw):
                if kw:
                    raise NotImplementedError("No support for keywords yet")

                pql_args = [objects.from_python(a) for a in args]
                pql_res = self.interp.call_func(fname, pql_args)
                return self._wrap_result( pql_res )
            return delegate
        else:
            obj = self.interp.evaluate_obj( var )
            return self._wrap_result(obj)

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        if isinstance(res, Ast):
            raise TypeError("Returned object cannot be converted into a Python representation")
        return _prepare_instance_for_user(self.interp, res)  # TODO session, not state

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


    def import_pandas(self, **dfs):
        """Import pandas.DataFrame instances into SQL tables

        Example:
            >>> pql.import_pandas(a=df_a, b=df_b)
        """
        return self.interp.import_pandas(dfs)


    def load_all_tables(self):
        return self.interp.load_all_tables()




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
