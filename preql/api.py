import logging
from pathlib import Path

from .interpreter import Interpreter
from . import pql_objects as pql
from . import ast_classes as ast
from .exceptions import PreqlError

class SqlEngine:
    pass

class SqliteEngine(SqlEngine):
    def __init__(self, filename=None, debug=True):
        import sqlite3
        self._conn = sqlite3.connect(filename or ':memory:')
        self._debug = debug

    def query(self, sql, qargs=(), quiet=False):
        c = self._conn.cursor()
        if self._debug and not quiet:
            for i, s in enumerate(sql.split('\n')):
                print('    ' if i else 'SQL>', s)
        c.execute(sql, qargs)
        return c.fetchall()

    def addmany(self, table, cols, values):
        assert all(len(v)==len(cols) for v in values)

        c = self._conn.cursor()
        qmarks = ','.join(['?'] * len(cols))
        cols_str = ','.join(cols)
        sql = f'INSERT INTO {table} ({cols_str}) VALUES ({qmarks})'
        if self._debug:
            print('SQL>', sql, end=' ')
        ids = []
        for v in values:
            c.execute(sql, v)
            ids.append(c.lastrowid)
        # c.executemany(sql, values)
        assert len(ids) == len(set(ids))
        inserted = len(ids)
        if self._debug:
            print('-- Inserted %d rows' % inserted)
        assert inserted == len(values)
        return ids


class RowWrapper:
    def __init__(self, row):
        self._row = row

    def __repr__(self):
        return self._row.repr()

    def __getitem__(self, item):
        return self._row.getattr(item)

    def __getattr__(self, attr):
        return self[attr]

    def __getstate__(self):
        return self._row
    def __setstate__(self, x):
        self._row = x


class TableWrapper:
    def __init__(self, pql_table, interp):
        self._pql_table = pql_table
        self._interp = interp

    def __repr__(self):
        return self._pql_table.repr(self._interp)

    def json(self):
        return [row.attrs for row in self._query()]

    def _query(self):
        return self._pql_table.query(self._interp, None)

    def __iter__(self):
        return (RowWrapper(row) for row in self._query())

    def __len__(self):
        return self._pql_table.count(self._interp).value
    

def python_to_pql(value):
    if value is None:
        return pql.null
    elif isinstance(value, str):
        return pql.String(value)
    elif isinstance(value, int):
        return pql.Integer(value)
    assert False, value

class Interface:
    def __init__(self, db_uri=None, debug=True):
        # TODO actually parse uri
        self.engine = SqliteEngine(db_uri, debug=debug)
        self.interp = Interpreter(self.engine)

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
        elif isinstance(res, pql.RowRef):
            return RowWrapper(res)
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

    def _functions(self):
        return {name:f for name,f in self.interp.state.namespace.items()
                if isinstance(f, ast.FunctionDef)}

    def add_many(self, table, values):
        cols = [c.name
                for c in self.interp.state.namespace[table].columns.values()
                if not isinstance(c.type, (ast.BackRefType, ast.IdType))]
        return self.interp.sqlengine.addmany(table, cols, values)

    def add(self, table, values):
        return self.add_many(table, [values])

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
                except PreqlError as e:
                    print(e)
                    continue
                except Exception as e:
                    print("Error:")
                    logging.exception(e)
                    continue


                if isinstance(res, pql.Object):
                    res = res.repr(self.interp) 

                # Print
                print(res)
        except (KeyboardInterrupt, EOFError):
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