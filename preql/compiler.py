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
        assert not issubclass(ast_type, Expr)
        return getattr(self, '_sql_' + ast_type.__name__)(ast_node)

    def _sqlexpr(self, ast_node, context=None):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr)
        return getattr(self, '_sqlexpr_' + ast_type.__name__)(ast_node, context)

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

    def _sqlexpr_Value(self, v: Value, context):
        assert v.type is StrType, v
        return '"%s"' % v.value

    def _sql_AddRow(self, addrow: AddRow):
        # cols, values = zip(*addrow.args)
        cols = [c.name for c in self.tables[addrow.table.name].columns[1:]]
        values = addrow.args
        q = ['INSERT INTO', addrow.table.name, 
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(self._sqlexpr(v) for v in values), ")",
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

    def _sqlexpr_Ref(self, ref: Ref, context):
        # if ref.name in context['params']:
        #     assert False
        name ,= ref.name

        try:
            v = context['args'][name]
            return self._sqlexpr(v)
        except KeyError:
            pass

        relation = context['relation']
        for c in relation.columns:
            if c.name == name:
                # return '%s.%s' % (relation.name)
                return name

        assert False, (ref, relation)

    def _sqlexpr_Compare(self, compare: Compare, context):
        elems = [self._sqlexpr(e, context) for e in compare.elems]
        return compare.op.join(elems)

    def _sqlexpr_Query(self, query: Query, context):
        assert query.as_ is None, query
        assert isinstance(query.relation, Ref), query
        table_name ,= query.relation.name   # Cannot handle join yet
        context = dict(context)
        context['relation'] = self.tables[table_name]
        sel_sql = ' AND '.join([
            self._sqlexpr(x, context)
            for x in query.selection
        ])
        assert query.groupby is None
        proj_sql = ', '.join([
            self._sqlexpr(x, context)
            for x in query.projection
        ])
        return f'SELECT {proj_sql} FROM {table_name} WHERE {sel_sql}'

        # selection: list
        # groupby: list
        # projection: list

    # def _sqlexpr_Function(self, func: Function, context):

    def _sqlexpr_FuncCall(self, funccall: FuncCall, context):
        f = self.functions[funccall.name]
        args = funccall.args
        assert len(args) == len(f.params or []), (args, f.params)
        args_d = dict(zip(f.params or [], args))
        return self._sqlexpr(f.expr, {'args': args_d})

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

    def compile_func_call(self, fname, args):
        print ('!!', fname, args)
        args = [Value.from_pyobj(a) for a in args]
        funccall = FuncCall(fname, args)
        return self._sqlexpr(funccall)

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