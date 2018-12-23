from .ast_classes import *
from .parser import parse, parse_query

def find_exprs(expr):
    for e in expr.exprs:
        yield e
        yield from find_exprs(e)


class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.tables = {}
        self.functions = {}
        self.vars = {}

    def _sql(self, ast_node):
        ast_type = type(ast_node)
        assert not issubclass(ast_type, Expr)
        return getattr(self, '_sql_' + ast_type.__name__)(ast_node)

    def _sqlexpr(self, ast_node, context=None):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr)
        return getattr(self, '_sqlexpr_' + ast_type.__name__)(ast_node, context)

    def _sql_Column(self, column: Column):
        mod = ' NOT NULL' if not column.is_nullable else ''
        mod += ' PRIMARY KEY' if column.is_pk else ''

        if isinstance(column.type, TableType):
            c = f'{column.name} INTEGER{mod}'
            fk = f'FOREIGN KEY({column.name}) REFERENCES {column.type.name}(id)'
            return c + ", " + fk
        else:
            sql_type = {
                IntType: 'INTEGER',
                StrType: 'VARCHAR(4000)',
                # 'real': 'REAL',
            }[column.type]
            return f'{column.name} {sql_type}{mod}'

    def _sql_Table(self, table: Table):
        return ' '.join(['CREATE TABLE', table.name, '(\n ', ',\n  '.join(
            self._sql(r) for r in table.columns
            ), '\n);'])

    def _sqlexpr_Value(self, v: Value, context):
        if v.type is IntType:
            return str(v.value)
        assert v.type is StrType, v
        return '"%s"' % v.value


    def _sqlexpr_Ref(self, ref: Ref, context):
        # if ref.name in context['params']:
        #     assert False
        name ,= ref.name

        try:
            return self._sqlexpr(self.vars[name], {})
        except KeyError:
            pass

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
        r = self.sqlengine.query( self._sql(table) )
        assert not r

    def _sql_AddRow(self, addrow: AddRow):
        cols = [c.name for c in self.tables[addrow.table.name].columns[1:]]
        values = addrow.args
        q = ['INSERT INTO', addrow.table.name, 
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(self._sqlexpr(v) for v in values), ")",
        ]
        insert = ' '.join(q) + ';'
        return insert

    def _add_row(self, addrow: AddRow):
        insert = self._sql(addrow)
        assert not self.sqlengine.query(insert)

        if addrow.as_:
            rowid ,= self.sqlengine.query('SELECT last_insert_rowid();')[0]
            self.vars[addrow.as_] = Value(IntType, rowid)   # TODO Id type?

    def _def_function(self, func: Function):
        assert func.name not in self.functions
        self.functions[func.name] = func
        return []



    def run_stmt(self, c):
        if isinstance(c, Table):
            self._add_table(c)
        # elif isinstance(c, Query):
        #     yield from self._compile_query(c)
        elif isinstance(c, Function):
            self._def_function(c)
        elif isinstance(c, AddRow):
            self._add_row(c)
        else:
            raise ValueError(c)

    def compile_query(self, query):
        return self._sql(query)

    # def compile_query(self, s):
    #     ast = parse_query(s)
    #     assert isinstance(ast, Query)
    #     sql ,= self._compile_query(ast)
    #     return sql
    def call_func(self, fname, args):
        args = [Value.from_pyobj(a) for a in args]
        funccall = FuncCall(fname, args)
        sql = self._sqlexpr(funccall)
        return self.sqlengine.query(sql)

    def execute(self, s):
        ast = parse(s)
        for stmt in ast:
            self.run_stmt(stmt)

    def query(self, q):
        ast = parse(q)
        sql = self.compile_query(ast)
        return self.sqlengine.query(sql)