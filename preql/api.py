from contextlib import contextmanager

import tabulate

from . import pql_ast as ast
from . import pql_types as types
from . import pql_objects as objects
from .interpreter import Interpreter
from .evaluate import localize, evaluate
from .interp_common import create_engine, call_pql_func


def _make_const(value):
    t = types.Primitive.by_pytype[type(value)]
    return ast.Const(None, t, value)

def _call_pql_func(state, name, args):
    count = call_pql_func(state, name, args)
    return localize(state, evaluate(state, count))

TABLE_PREVIEW_SIZE = 10
MAX_AUTO_COUNT = 10000

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
            return call_pql_func(self._state, 'limit_offset', [self._inst, _make_const(limit), _make_const(offset)])

        res ,= localize(self._state, evaluate(self._state, self[index:1]))
        return res

    def __repr__(self):
        count = _call_pql_func(self._state, 'count', [self[:MAX_AUTO_COUNT]])
        if count == MAX_AUTO_COUNT:
            count_str = f'count>={count}'
        else:
            count_str = f'count={count}'

        rows = list(_call_pql_func(self._state, 'limit', [self._inst, ast.Const(None, types.Int, TABLE_PREVIEW_SIZE)]))
        post = '\n\t...' if len(rows) < count else ''

        if self._state.fmt == 'html':
            header = f"<pre>table {self._inst.type.name}, {count_str}</pre>"
            if rows:
                cols = list(rows[0])
                ths = '<tr>%s</tr>' % ' '.join([f"<th>{col}</th>" for col in cols])
                trs = [
                    '<tr>%s</tr>' % ' '.join([f"<td>{v}</td>" for v in row.values()])
                    for row in rows
                ]

            return '%s<table>%s%s</table>' % (header, ths, '\n'.join(trs)) + post
        else:
            header = f"table {self._inst.type.name}, {count_str}\n"
            return header + tabulate.tabulate(rows, headers="keys", numalign="right") + post


def promise(state, inst):
    if isinstance(inst, objects.TableInstance):
        if not isinstance(inst.type, types.ListType):
            return TablePromise(state, inst)

    return localize(state, inst)


class Interface:
    __name__ = "Preql"

    def __init__(self, db_uri=None, debug=True, save_last=None):
        if db_uri is None:
            db_uri = 'sqlite://:memory:'

        self.engine = create_engine(db_uri, debug=debug)
        self.interp = Interpreter(self.engine)
        self.save_last = save_last

        # self.interp.state._py_api = self # TODO proper api

    def exec(self, q, *args, **kw):
        "Deprecated"
        return self.interp.execute_code(q, *args, **kw)

    def close(self):
        self.engine.close()

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
        assert not isinstance(res, ast.Ast)
        return promise(self.interp.state, res)  # TODO session, not state


    def __call__(self, pq, **args):
        pql_args = {name: objects.from_python(value) for name, value in args.items()}

        res = self.interp.execute_code(pq + "\n", pql_args)
        if res:
            if self.save_last:
                self.interp.set_var(self.save_last, res)

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

    def start_repl(self):
        from .repl import start_repl
        start_repl(self)

    def commit(self):
        return self.engine.commit()

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


#     def start_repl(self):
#         from prompt_toolkit import prompt
#         from prompt_toolkit import PromptSession
#         from pygments.lexers.python import Python3Lexer
#         from prompt_toolkit.lexers import PygmentsLexer

#         try:
#             session = PromptSession()
#             while True:
#                 # Read
#                 code = session.prompt(' >> ', lexer=PygmentsLexer(Python3Lexer))
#                 if not code.strip():
#                     continue

#                 # Evaluate
#                 try:
#                     res = self(code)
#                 except PreqlError as e:
#                     print(e)
#                     continue
#                 except Exception as e:
#                     print("Error:")
#                     logging.exception(e)
#                     continue


#                 if isinstance(res, pql.Object):
#                     res = res.repr(self.interp)

#                 # Print
#                 print(res)
#         except (KeyboardInterrupt, EOFError):
#             print('Exiting Preql interaction')


