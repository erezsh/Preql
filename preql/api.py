from contextlib import contextmanager
from functools import wraps

from . import settings
from .core.pql_ast import pyvalue, Ast
from .core import pql_objects as objects
from .core.interpreter import Interpreter
from .core.pql_types import T
from .sql_interface import create_engine
from .utils import dsp
from .core.exceptions import Signal

from .core import display
display.install_reprs()



def clean_signal(f):
    @wraps(f)
    def inner(*args, **kwargs):
        if settings.debug:
            return f(*args, **kwargs)

        try:
            return f(*args, **kwargs)
        except Signal as e:
            raise e.clean_copy() from None  # Error from Preql
    return inner


class TablePromise:
    """Returned by Preql whenever the result is a table

    Fetching values creates queries to database engine
    """

    def __init__(self, interp, inst):
        self._interp = interp
        self._inst = inst
        self._rows = None

    @property
    def type(self):
        return self._inst.type

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
            limit = (index.stop or len(self)) - offset
            new_inst = self._interp.call_builtin_func('limit_offset', [self._inst, pyvalue(limit), pyvalue(offset)])
            return TablePromise(self._interp, new_inst)

        # TODO different debug log level / mode
        res ,= self._interp.cast_to_python(self[index:index+1]._inst)
        return res

    def __repr__(self):
        with self._interp.setup_context():
            return display.print_to_string(display.table_repr(self._inst), 'text')



@dsp
def from_python(value: TablePromise):
    return value._inst


def _prepare_instance_for_user(interp, inst):
    if inst.type <= T.table:
        return TablePromise(interp, inst)

    return interp.localize_obj(inst)


class _Delegate:

    def __init__(self, pql, fname):
        self.fname = fname
        self.pql = pql

    @clean_signal
    def __call__(self, *args, **kw):
        pql_args = [objects.from_python(a) for a in args]
        pql_kwargs = {k:objects.from_python(v) for k,v in kw.items()}
        pql_res = self.pql._interp.call_func(self.fname, pql_args, pql_kwargs)
        return self.pql._wrap_result( pql_res )


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

    def __repr__(self):
        return f'Preql({self._db_uri!r}, ...)'

    def __getstate__(self):
        return self._db_uri, self._print_sql, self._display, self._interp

    def set_output_format(self, fmt):
        if fmt == 'html':
            self._display = display.HtmlDisplay()
        else:
            self._display = display.RichDisplay()

        self._interp.state.state.display = self._display  # TODO proper api


    def _reset_interpreter(self, engine=None):
        if engine is None:
            engine = self._interp.state.db
        self._interp = Interpreter(engine, self._display)
        self._interp._py_api = self # TODO proper api

    def close(self):
        self._interp.state.db.close()

    def __getattr__(self, fname):
        var = self._interp.state.get_var(fname)

        if isinstance(var, objects.Function):
            return _Delegate(self, fname)
            # @clean_signal
            # def delegate(*args, **kw):
            #     pql_args = [objects.from_python(a) for a in args]
            #     pql_kwargs = {k:objects.from_python(v) for k,v in kw.items()}
            #     pql_res = self._interp.call_func(fname, pql_args, pql_kwargs)
            #     return self._wrap_result( pql_res )
            # return delegate
        else:
            obj = self._interp.evaluate_obj( var )
            return self._wrap_result(obj)

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        if isinstance(res, Ast):
            raise TypeError("Returned object cannot be converted into a Python representation")
        return _prepare_instance_for_user(self._interp, res)  # TODO session, not state

    def _run_code(self, code, source_name='<api>', args=None):
        pql_args = {name: objects.from_python(value) for name, value in (args or {}).items()}
        return self._interp.execute_code(code + "\n", source_name, pql_args)

    @clean_signal
    def __call__(self, code, **args):
        res = self._run_code(code, '<inline>', args)
        if res:
            return self._wrap_result(res)

    @clean_signal
    def load(self, filename, rel_to=None):
        """Load a Preql script

        Parameters:
            filename (str): Name of script to run
            rel_to (Optional[str]): Path to which ``filename`` is relative.
        """
        self._interp.include(filename, rel_to)

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
        return self._interp.state.db.commit()

    def rollback(self):
        return self._interp.state.db.rollback()


    def import_pandas(self, **dfs):
        """Import pandas.DataFrame instances into SQL tables

        Example:
            >>> pql.import_pandas(a=df_a, b=df_b)
        """
        return self._interp.import_pandas(dfs)


    def load_all_tables(self):
        return self._interp.load_all_tables()

    @property
    def interp(self):
        raise Exception("Reserved")
    




#     def _functions(self):
#         return {name:f for name,f in self._interp.state.namespace.items()
#                 if isinstance(f, ast.FunctionDef)}

#     def add_many(self, table, values):
#         cols = [c.name
#                 for c in self._interp.state.namespace[table].columns.values()
#                 if not isinstance(c.type, (ast.BackRefType, ast.IdType))]
#         return self.engine.addmany(table, cols, values)

#     def add(self, table, values):
#         return self.add_many(table, [values])
