from pathlib import Path

from .sql_interface import SqliteInterface
from . import pql_types as types
from . import pql_ast as ast
from . import pql_objects as objects
from .interpreter import Interpreter
from .evaluate import sql_repr

def python_to_pql(value):
    if value is None:
        return types.null
    elif isinstance(value, str):
        return ast.Const(types.String, value)
    elif isinstance(value, int):
        return ast.Const(types.Int, value)
    assert False, value


class Interface:
    def __init__(self, db_uri=None, debug=True):
        # TODO actually parse uri
        self.engine = SqliteInterface(db_uri, debug=debug)
        self.interp = Interpreter(self.engine)

    def exec(self, q, *args, **kw):
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
        return res


    def __call__(self, pq, **args):
        pql_args = {name: python_to_pql(value) for name, value in args.items()}

        res = self.interp.eval_expr(pq, pql_args)
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


