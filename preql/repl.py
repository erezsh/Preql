import sys
import logging
from pathlib import Path

from . import Preql
from . import pql_types as types
from . import pql_ast as ast
from .exceptions import PreqlError


class RowWrapper:
    def __init__(self, row):
        self._row = row

    def __repr__(self):
        return self._row.repr()

    def __getitem__(self, item):
        return self._row.getattr(item)

    def __getattr__(self, attr):
        return self[attr]

    def __iter__(self):
        return iter(self._row)

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


from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from pygments.lexers.python import Python3Lexer
from prompt_toolkit.lexers import PygmentsLexer

def start_repl(p):

    try:
        session = PromptSession()
        while True:
            # Read
            # code = session.prompt(' >> ', lexer=PygmentsLexer(Python3Lexer))
            code = session.prompt(' >> ')
            if not code.strip():
                continue

            # Evaluate
            try:
                res = p(code)
            except PreqlError as e:
                if e.meta:
                    print(f"Error at line {e.meta.start_line, e.meta.start_column}: {e.message}")
                    print()
                    print(e.get_context(code))
                else:
                    print(e)
                continue
            except Exception as e:
                print("Error:")
                logging.exception(e)
                raise
                # continue


            if isinstance(res, types.PqlObject):
                res = res.repr(p.interp)

            # Print
            print(res)
    except (KeyboardInterrupt, EOFError):
        print('Exiting Preql interaction')


def main(db=None):
    p = Preql(db)
    start_repl(p)

if __name__ == '__main__':
    main(*sys.argv[1:])