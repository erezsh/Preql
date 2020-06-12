from contextlib import contextmanager

import tabulate

from . import settings
from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects
from .interpreter import Interpreter
from .evaluate import localize, evaluate, new_table_from_rows
from .interp_common import create_engine, call_pql_func, State, from_python
from .pql_types import T


def _make_const(value):
    # t = types.Primitive.by_pytype[type(value)]
    t = types.from_python(type(value))
    return ast.Const(None, t, value)

def _call_pql_func(state, name, args):
    count = call_pql_func(state, name, args)
    return localize(state, evaluate(state, count))

TABLE_PREVIEW_SIZE = 10
LIST_PREVIEW_SIZE = 100
MAX_AUTO_COUNT = 10000



def table_limit(self, state, limit):
    return call_pql_func(state, '_core_limit', [self, _make_const(limit)])

def table_repr(self, state):

    assert isinstance(state, State), state
    count = _call_pql_func(state, 'count', [table_limit(self, state, MAX_AUTO_COUNT)])
    if count == MAX_AUTO_COUNT:
        count_str = f'>={count}'
    else:
        count_str = f'={count}'

    if len(self.type.elems) == 1:
        rows = localize(state, table_limit(self, state, LIST_PREVIEW_SIZE))
        post = f', ... ({count_str})' if len(rows) < count else ''
        elems = ', '.join(repr(r) for r in rows)
        return f'[{elems}{post}]'

    # rows = list(_call_pql_func(state, 'limit', [self, _make_const(TABLE_PREVIEW_SIZE)]))
    rows = localize(state, table_limit(self, state, TABLE_PREVIEW_SIZE))
    if (self.type <= T.list):
        rows = [{'value': x} for x in rows]

    post = '\n\t...' if len(rows) < count else ''

    if state.fmt == 'html':
        header = f"<pre>table {self.type.name}, {count_str}</pre>"
        if rows:
            cols = list(rows[0])
            ths = '<tr>%s</tr>' % ' '.join([f"<th>{col}</th>" for col in cols])
            trs = [
                '<tr>%s</tr>' % ' '.join([f"<td>{v}</td>" for v in row.values()])
                for row in rows
            ]

        return '%s<table>%s%s</table>' % (header, ths, '\n'.join(trs)) + post
    else:
        header = f"table {self.type.name}, {count_str}\n"
        return header + tabulate.tabulate(rows, headers="keys", numalign="right") + post

objects.TableInstance.repr = table_repr


class TablePromise:
    def __init__(self, state, inst):
        self._state = state
        self._inst = inst
        self._rows = None

    def to_json(self):
        if self._rows is None:
            self._rows = localize(self._state, self._inst)
        assert self._rows is not None
        return self._rows

    def to_pandas(self):
        from pandas import DataFrame
        return DataFrame(self)

    def __eq__(self, other):
        return self.to_json() == other

    def __len__(self):
        return _call_pql_func(self._state, 'count', [self._inst])

    def __iter__(self):
        return iter(self.to_json())

    def __getitem__(self, index):
        if isinstance(index, slice):
            offset = index.start or 0
            limit = index.stop - offset
            return call_pql_func(self._state, '_core_limit_offset', [self._inst, _make_const(limit), _make_const(offset)])

        res ,= localize(self._state, evaluate(self._state, self[index:1]))
        return res

    def __repr__(self):
        return self._inst.repr(self._state)


def promise(state, inst):
    if (inst.type <= T.collection):
        if not (inst.type <= T.list):
            return TablePromise(state, inst)

    return localize(state, inst)


class Interface:
    __name__ = "Preql"

    def __init__(self, db_uri=None, debug=settings.debug, save_last=None):
        if db_uri is None:
            db_uri = 'sqlite://:memory:'

        self.engine = create_engine(db_uri, debug=debug)
        self.interp = Interpreter(self.engine)

        self.interp.state._py_api = self # TODO proper api

    def close(self):
        self.engine.close()

    def __getattr__(self, fname):
        var = self.interp.state.get_var(fname)
        if isinstance(var, objects.Function):
            def delegate(*args, **kw):
                assert not kw
                pql_args = [from_python(a) for a in args]
                pql_res = self.interp.call_func(fname, pql_args)
                return self._wrap_result( pql_res )
            return delegate
        else:
            return self._wrap_result( evaluate( self.interp.state, var ))

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        assert not isinstance(res, ast.Ast), res
        return promise(self.interp.state, res)  # TODO session, not state

    def run_code(self, pq, source_file, **args):
        pql_args = {name: from_python(value) for name, value in args.items()}
        return self.interp.execute_code(pq + "\n", source_file, pql_args)

    def __call__(self, pq, **args):
        res = self.run_code(pq, '<inline>', **args)
        if res:
            return self._wrap_result(res)

    def load(self, fn, rel_to=None):
        self.interp.include(fn, rel_to)

    @contextmanager
    def transaction(self):
        # TODO rollback
        try:
            yield self  # TODO new instance?
        finally:
            self.commit()

    def start_repl(self, *args):
        from .repl import start_repl
        start_repl(self, *args)

    def commit(self):
        return self.engine.commit()

    def import_pandas(self, **dfs):
        for name, df in dfs.items():
            cols = list(df)
            rows = [[i.item() if hasattr(i, 'item') else i for i in rec]
                    for rec in df.to_records()]
            new_table_from_rows(self.interp.state, name, cols, rows)







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
