import logging
from pathlib import Path

from .interpreter import Interpreter
from . import pql_objects as pql
from . import ast_classes as ast

class SqlEngine:
    pass

class SqliteEngine(SqlEngine):
    def __init__(self, filename=None):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')

    def query(self, sql):
        dargs = {}
        c = self._conn.cursor()
        for i, s in enumerate(sql.split('\n')):
            print('    ' if i else 'SQL>', s)
        c.execute(sql, dargs)
        return c.fetchall()


class RowWrapper:
    def __init__(self, row):
        self._row = row

    def __repr__(self):
        return self._row.repr()

class TableWrapper:
    def __init__(self, pql_table, interp):
        self._pql_table = pql_table
        self._interp = interp

    def __repr__(self):
        return self._pql_table.repr(self._interp)

    def _query(self):
        return self._pql_table.query(self._interp, None)

    def __iter__(self):
        return (RowWrapper(row) for row in self._query())
    


class Interface:
    def __init__(self, db_uri=None):
        # TODO actually parse uri
        self.interp = Interpreter(SqliteEngine(db_uri))

    def exec(self, q, *args, **kw):
        return self.interp.execute_code(q, *args, **kw)

    def __getattr__(self, fname):
        def delegate(*args, **kw):
            assert not kw
            return self._wrap_result( self.interp.call_func(fname, args) )
        return delegate

    def _wrap_result(self, res):
        "Wraps Preql result in a Python-friendly object"
        if isinstance(res, pql.Table):
            return TableWrapper(res, self.interp)
        return res


    def __call__(self, pq):
        res = self.interp.eval_expr(pq)
        return self._wrap_result(res)

    def load(self, fn, rel_to=None):
        """Load content filename as Preql code

        If rel_to is provided, the function will find the filename in relation to it.
        """
        if rel_to:
            fn = Path(rel_to).parent / fn
        with open(fn, encoding='utf8') as f:
            self.exec(f.read())

    def _functions(self):
        return {name:f for name,f in self.interp.state.namespace.items()
                if isinstance(f, ast.FunctionDef)}

    def start_repl(self):
        from prompt_toolkit import prompt
        from prompt_toolkit import PromptSession
        from pygments.lexers.python import Python3Lexer
        from prompt_toolkit.lexers import PygmentsLexer

        try:
            session = PromptSession()
            while True:
                # Read
                code = session.prompt(' >> ', lexer=PygmentsLexer(Python3Lexer))
                if not code.strip():
                    continue

                # Evaluate
                try:
                    res = self(code)
                except Exception as e:
                    print("Error:")
                    logging.exception(e)
                    continue

                if isinstance(res, pql.Object):
                    res = res.repr(self.interp) 

                # Print
                print(res)
        except KeyboardInterrupt:
            print('Exiting Preql interaction')




def test1():
    i = Interface()
    i.load('simple1.pql', rel_to=__file__)
    print(i.english())
    print(i.by_country('Israel'))
    print(i.english2())

def test2():
    i = Interface()
    i.load('simple2.pql', rel_to=__file__)
    # print(i.english_speakers())
    # print(i.person_and_language())
    # print(i.from_my_country())
    # print(i.population_count())
    # print(i.citizens_list())
    # print(i.explicit_join2())
    print(i.explicit_join())

def test3():
    a = open("preql/tree.pql").read()
    i = Interface()
    i(a)
    lion = i.lion()
    print(lion)
    print(i.up(lion['id']))

    # i.up() - requires join aliases!
    # i.animals() - Requires advanced type system

# print('---------')
# test1()
# print('---------')
# test2()