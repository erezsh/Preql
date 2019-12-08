from pathlib import Path

from .sql_interface import SqliteInterface
from . import pql_types as types
from . import pql_ast as ast
from . import pql_objects as objects
from .interpreter import Interpreter
from .evaluate import localize

def python_to_pql(value):
    if value is None:
        return types.null
    elif isinstance(value, str):
        return ast.Const(types.String, value)
    elif isinstance(value, int):
        return ast.Const(types.Int, value)
    assert False, value



TABLE_PREVIEW_SIZE = 20

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
        return len(self.to_json())  # TODO use pql_count

    def __iter__(self):
        return iter(self.to_json())

    def __getitem__(self, index):
        return self.to_json()[index]

    def __repr__(self):
        count = len(self)
        header = f"<pre>table {self._inst.type.name}, count={count}</pre>"
        rows = self[:TABLE_PREVIEW_SIZE]    # TODO const
        if rows:
            cols = list(rows[0])
            ths = '<tr>%s</tr>' % ' '.join([f"<th>{col}</th>" for col in cols])
            trs = [
                '<tr>%s</tr>' % ' '.join([f"<td>{v}</td>" for v in row.values()])
                for row in rows
            ]

        return '%s<table>%s%s</table>' % (header, ths, '\n'.join(trs))


def promise(state, inst):
    if isinstance(inst, objects.TableInstance):
        if not isinstance(inst.type, types.ListType):
            return TablePromise(state, inst)

    return localize(state, inst)


class Interface:
    def __init__(self, db_uri=None, debug=True):
        # TODO actually parse uri
        self.engine = SqliteInterface(db_uri, debug=debug)
        self.interp = Interpreter(self.engine)

    def exec(self, q, *args, **kw):
        "Deprecated"
        return self.interp.execute_code(q, *args, **kw)

    def __getattr__(self, fname):
        def delegate(*args, **kw):
            assert not kw
            return self._wrap_result( self.interp.call_func(fname, [python_to_pql(a) for a in args]) )
        return delegate

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        # if isinstance(res, pql.Table):
        #     return TableWrapper(res, self.interp)
        # elif isinstance(res, pql.RowRef):
        #     return RowWrapper(res)
        return promise(self.interp.state, res)  # TODO session, not state


    def __call__(self, pq, **args):
        pql_args = {name: python_to_pql(value) for name, value in args.items()}

        res = self.interp.execute_code(pq + "\n", pql_args)
        return self._wrap_result(res)

    def load(self, fn, rel_to=None):
        """Load content filename as Preql code

        If rel_to is provided, the function will find the filename in relation to it.
        """
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self.exec(f.read())

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

#     def commit(self):
#         return self.engine.commit()

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

