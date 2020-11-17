from contextlib import contextmanager
from datetime import datetime
from rich import table

import rich.table
import rich.markup

from . import settings
from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects
from . import exceptions as exc
from .utils import classify
from .interpreter import Interpreter
from .evaluate import cast_to_python, localize, evaluate, new_table_from_rows
from .interp_common import create_engine, call_pql_func, State
from .pql_types import T, ITEM_NAME
from .exceptions import Signal



def _make_const(value):
    # t = types.Primitive.by_pytype[type(value)]
    t = types.from_python(type(value))
    return ast.Const(t, value)

def _call_pql_func(state, name, args):
    count = call_pql_func(state, name, args)
    return cast_to_python(state, count)

TABLE_PREVIEW_SIZE = 16
LIST_PREVIEW_SIZE = 128
MAX_AUTO_COUNT = 10000



def table_limit(self, state, limit, offset=0):
    return call_pql_func(state, 'limit_offset', [self, _make_const(limit), _make_const(offset)])


def _html_table(name, count_str, rows, offset):
    header = 'table '
    if name:
        header += name
    if offset:
        header += f'[{offset}..]'
    header += f" {count_str}"
    header = f"<pre>table {name}, {count_str}</pre>"

    if not rows:
        return header

    cols = list(rows[0])
    ths = '<tr>%s</tr>' % ' '.join([f"<th>{col}</th>" for col in cols])
    trs = [
        '<tr>%s</tr>' % ' '.join([f"<td>{v}</td>" for v in row.values()])
        for row in rows
    ]

    return '%s<table>%s%s</table>' % (header, ths, '\n'.join(trs))


def _rich_table(name, count_str, rows, offset, has_more, colors=True, show_footer=False):
    header = 'table '
    if name:
        header += name
    if offset:
        header += f'[{offset}..]'
    header += f" {count_str}"

    if not rows:
        return header

    table = rich.table.Table(title=rich.markup.escape(header), show_footer=show_footer)

    # TODO enable/disable styling
    for k, v in rows[0].items():
        kw = {}
        if isinstance(v, (int, float)):
            kw['justify']='right'

        if colors:
            if isinstance(v, int):
                kw['style']='cyan'
            elif isinstance(v, float):
                kw['style']='yellow'
            elif isinstance(v, str):
                kw['style']='green'

        table.add_column(k, footer=k, **kw)

    for r in rows:
        table.add_row(*[rich.markup.escape(str(x) if x is not None else '-') for x in r.values()])

    if has_more:
        table.add_row(*['...' for x in rows[0]])

    return table


_g_last_table = None
_g_last_offset = 0
def table_more(state):
    if not _g_last_table:
        raise Signal.make(T.ValueError, state, None, "No table yet")

    return table_repr(_g_last_table, state, _g_last_offset)


def table_repr(self, state, offset=0):
    global _g_last_table, _g_last_offset

    assert isinstance(state, State), state
    count = _call_pql_func(state, 'count', [table_limit(self, state, MAX_AUTO_COUNT)])
    if count == MAX_AUTO_COUNT:
        count_str = f'>={count}'
    else:
        count_str = f'={count}'

    # if len(self.type.elems) == 1:
    #     rows = cast_to_python(state, table_limit(self, state, LIST_PREVIEW_SIZE))
    #     post = f', ... ({count_str})' if len(rows) < count else ''
    #     elems = ', '.join(repr_value(ast.Const(None, self.type.elem, r)) for r in rows)
    #     return f'[{elems}{post}]'

    # TODO load into preql and repr, instead of casting to python
    rows = cast_to_python(state, table_limit(self, state, TABLE_PREVIEW_SIZE, offset))
    _g_last_table = self
    _g_last_offset = offset + len(rows)
    if self.type <= T.list:
        rows = [{ITEM_NAME: x} for x in rows]

    has_more = offset + len(rows) < count

    try:
        table_name = self.type.options['name'].repr_name
    except KeyError:
        table_name = ''

    if state.fmt == 'html':
        return _html_table(table_name, count_str, rows, offset)
    elif state.fmt == 'rich':
        return _rich_table(table_name, count_str, rows, offset, has_more)

    assert state.fmt == 'text'
    return _rich_table(table_name, count_str, rows, offset, has_more, colors=False)

    # raise NotImplementedError(f"Unknown format: {state.fmt}")

objects.CollectionInstance.repr = table_repr


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
        if isinstance(index, slice):
            offset = index.start or 0
            limit = index.stop - offset
            return call_pql_func(self._state, 'limit_offset', [self._inst, _make_const(limit), _make_const(offset)])

        # TODO different debug log level / mode
        # inst = evaluate(self._state,
        res ,= cast_to_python(self._state, self[index:index+1])
        return res

    def __repr__(self):
        return repr(self.to_json()) #str(self._inst.repr(self._state))


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

    def __init__(self, db_uri: str='sqlite://:memory:', print_sql: bool=settings.print_sql):
        """Initialize a new Preql instance

        Parameters:
            db_uri (str, optional): URI of database. Defaults to using a non-persistent memory database.
            print_sql (bool, optional): Whether or not to print every SQL query that is executed (default defined in settings)
        """
        self._db_uri = db_uri
        self._print_sql = print_sql
        # self.engine.ping()

        engine = create_engine(self._db_uri, print_sql=self._print_sql)
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
        # TODO rollback
        try:
            yield self  # TODO new instance?
        finally:
            self.commit()

    def start_repl(self, *args):
        "Run the interactive prompt"
        from .repl import start_repl
        start_repl(self, *args)

    def commit(self):
        return self.interp.state.db.commit()

    def _drop_tables(self, *tables):
        # XXX temporary method
        for t in tables:
            self.interp.state.db._execute_sql(T.nulltype, f"DROP TABLE {t};", self.interp.state)

    def import_pandas(self, **dfs):
        """Import pandas.DataFrame instances into SQL tables

        Example:
            >>> pql.import_pandas(a=df_a, b=df_b)
        """
        import pandas as pd
        def normalize_item(i):
            if pd.isna(i):
                return None
            i = i.item() if hasattr(i, 'item') else i
            return i

        for name, df in dfs.items():
            if isinstance(df, pd.Series):
                cols = ['key', 'value']
                rows = [(dt.to_pydatetime() if isinstance(dt, datetime) else dt,v) for dt, v in df.items()]
            else:
                assert isinstance(df, pd.DataFrame)
                cols = list(df)
                rows = [[normalize_item(i) for i in rec]
                        for rec in df.to_records()]
                rows = [ row[1:] for row in rows ]    # drop index

            new_table_from_rows(self.interp.state, name, cols, rows)

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
