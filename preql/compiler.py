from .ast_classes import *
from .parser import parse, parse_query

def find_exprs(expr):
    for e in expr.exprs:
        yield e
        yield from find_exprs(e)


class Compiler:
    def __init__(self):
        self.tables = {}
        self.functions = {}

        self.added_variables = False

    def _sql(self, ast_node):
        ast_type = type(ast_node)
        return getattr(self, '_sql_' + ast_type.__name__)(ast_node)

    def _sql_Column(self, column: Column):
        sql_type = {
            IntType: 'INTEGER',
            StrType: 'VARCHAR(4000)',
            # 'real': 'REAL',
        }[column.type]
        mod = ' NOT NULL' if not column.is_nullable else ''
        mod += ' PRIMARY KEY' if column.is_pk else ''
        return f'{column.name} {sql_type}{mod}'

    def _sql_Table(self, table: Table):
        yield ' '.join(['CREATE TABLE', table.name, '(\n ', ',\n  '.join(
            self._sql(r) for r in table.columns
            ), '\n);'])

    def _sql_Value(self, v: Value):
        assert v.type is StrType, v
        return '"%s"' % v.value

    def _sql_AddRow(self, addrow: AddRow):
        cols, values = zip(*addrow.assigns)
        q = ['INSERT INTO', addrow.table.name, 
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(self._sql(v) for v in values), ")",
        ]
        insert = ' '.join(q) + ';'

        if addrow.as_:
            set_var = AddRow('_variables', {
                'name': Value('"%s"' % addrow.as_),
                'rowid': Value('last_insert_rowid()'),
            }.items(), None)
            assert set_var.as_ is None
            yield from self._require_variables()
            yield insert
            yield from self._sql(set_var)
        else:
            yield insert

    def _sql_Function(self, func: Function):
        print('@@', func.query)
        import pdb
        pdb.set_trace()
        return []


    def _add_table(self, table):
        self.tables[table.name] = table
        return self._sql(table)

    def _add_row(self, table):
        return self._sql(table)


    def _def_function(self, func: Function):
        assert func.name not in self.functions
        self.functions[func.name] = func
        return []



    def compile_ast(self, commands):
        for c in commands:
            if isinstance(c, Table):
                yield from self._add_table(c)
            # elif isinstance(c, Query):
            #     yield from self._compile_query(c)
            elif isinstance(c, Function):
                yield from self._def_function(c)
            elif isinstance(c, AddRow):
                yield from self._add_row(c)
            else:
                raise ValueError(c)

    def compile_func_call(self, fname):
        f = self.functions[fname]
        return self._sql(f), f.params

    def compile_query(self, query):
        return self._sql(query)

    # def compile_query(self, s):
    #     ast = parse_query(s)
    #     assert isinstance(ast, Query)
    #     sql ,= self._compile_query(ast)
    #     return sql

    def compile_statements(self, s):
        ast = parse(s)
        return self.compile_ast(ast)

def test():
    a = open("preql/simple1.pql").read()
    c = Compiler()
    print('\n'.join(c.compile_statements(a)))

test()