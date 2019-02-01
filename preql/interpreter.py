from copy import copy, deepcopy
from contextlib import suppress

from .ast_classes import *
from .parser import parse, parse_query


@dataclass
class CompiledSQL:
    sql: str
    types: list
    names: list = None


class Context(list):
    def get(self, name):
        for d in self[::-1]:
            if name in d:
                return d[name]
        raise KeyError(name)

@dataclass
class State:
    functions = {}
    tables = {}
    vars = {}
    

# class ResolveTables:
#     def __init__(self, state):
#         self.state = state

#     def _resolve_tables(self, ast_node):
#         ast_type = type(ast_node)
#         assert issubclass(ast_type, Expr), (ast_node)
#         resolve_f = getattr(self, ast_type.__name__)
#         if resolve_f is NotImplemented:
#             return []
#         return resolve_f(ast_node)

#     def resolve(self, expr):
#         return self._resolve_tables(expr)


class ResolveIdentifiers:

    def __init__(self, state):
        self.state = state
        # self.functions = interp.functions
        # self.vars = interp.vars
        # self.tables = interp.tables
        self.context = Context()

    Value = NotImplemented
    
    def _resolve_exprs(self, expr: Expr):
        return self._resolve_expr_list(expr.exprs)

    Arith = _resolve_exprs
    Compare = _resolve_exprs

    def _resolve_expr_list(self, exprs):
        return [self._resolve_expr(e) for e in exprs]

    def _resolve_expr(self, ast_node):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr), (ast_node)
        resolve_f = getattr(self, ast_type.__name__)
        if resolve_f is NotImplemented:
            return
        return resolve_f(ast_node)

    # def _resolve_tables(self, ast_node):
    #     raise Exception()
    #     ast_type = type(ast_node)
    #     assert issubclass(ast_type, Expr), (ast_node)
    #     resolve_f = getattr(self, 'tables_' + ast_type.__name__)
    #     # if resolve_f is NotImplemented:
    #     #     return []
    #     return resolve_f(ast_node)

    def resolve(self, expr):
        return self._resolve_expr(expr)

    def Projection(self, proj: Projection):
        self._resolve_expr(proj.table)
        proj.resolved_table = proj.table.resolved_table
        assert proj.resolved_table, proj
        self.context.append({'table': proj.resolved_table})
        self._resolve_exprs(proj)
        self.context.pop()
        # TODO return projected table

    def Selection(self, sel: Selection):
        self._resolve_expr(sel.table)
        sel.resolved_table = sel.table.resolved_table
        assert sel.resolved_table, sel
        self.context.append({'table': sel.resolved_table})
        self._resolve_exprs(sel)
        self.context.pop()

    def Identifier(self, ident: Identifier):
        assert not ident.resolved

        basename = ident.name[0]
        # assert not ident.name[1:]   # TODO

        args = self.context.get('args')

        path = ident.name[1:]
        if basename in args:
            obj = args[basename]
        elif basename in self.state.vars:
            assert False
        elif basename in self.state.tables: 
            obj = self.state.tables[basename]
        else:
            # Check if basename is a column in the current table
            obj = self.context.get('table')
            path = ident.name
            assert obj

        
        for name in path:
            assert isinstance(obj.type, TabularType)

            obj = obj.resolved_table[name]

            if isinstance(obj.type, RelationalType):
                assert False
            elif isinstance(obj.type, TabularType):
                pass
            else:
                assert isinstance(obj.type, AtomType), obj
                pass


        # if len(ident.name)>1:
        #     table = self.context.get('table')
        #     print('@@', table)
        #     print('@@', table[basename])
        #     import pdb
        #     pdb.set_trace()
        
        ident.resolved = obj
        return ident.resolved

    def NamedExpr(self, named_expr):
        self._resolve_expr(named_expr.expr)

    def AutoJoin(self, join: AutoJoin):
        self._resolve_expr_list(join.exprs)
        join.resolved_table = join

    def AliasedTable(self, aliased):
        # TODO clunky code. Put into generic methods
        self._resolve_expr(aliased.table)
        resolved = aliased.table.resolved_table 
        columns = [copy(c) for c in resolved.columns]
        renamed_table = NamedTable(columns, aliased.name)
        for c in columns:
            c.table = renamed_table
        aliased.resolved_table = renamed_table
        
    def FuncCall(self, call: FuncCall):
        if call.name in ('join', 'freejoin'):
            assert not call.args.pos_args

            cls = {
                'join': AutoJoin,
                'freejoin': FreeJoin,
            }[call.name]

            args = []
            for name, table in call.args.named_args.items():
                new_table = AliasedTable(table, name)
                args.append(new_table)

            join = cls(args)
            call.resolved = join
            return self._resolve_expr(join)

        assert not call.args.named_args # TODO

        if call.name in ('to_array', 'count'):
            args = call.args.pos_args
            self._resolve_expr_list(args)
            if call.name == 'count':
                call.resolved = Count(args)
            else:
                assert False, call
            return


        f = self.state.functions[call.name]
        expr = deepcopy(f.expr)

        args_d = dict(zip(f.params or [], call.args.pos_args))

        assert call.name not in args_d
        self.context.append( {'args': args_d, 'func': f} )
        try:
            self._resolve_expr_list(call.args.pos_args)
            table = self._resolve_expr(expr)
        finally:
            self.context.pop()

        call.resolved = expr
        return table


class CompileSQL_Stmts:
    def __init__(self, state):
        self.state = state

    def to_sql(self, ast_node):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Stmt), ast_node
        return getattr(self, ast_type.__name__)(ast_node)

    def Column(self, column: Column):
        mod = ' NOT NULL' if not column.is_nullable else ''
        mod += ' PRIMARY KEY' if column.is_pk else ''

        if isinstance(column.type, RelationalType):
            c = f'{column.name} INTEGER{mod}'
            fk = f'FOREIGN KEY({column.name}) REFERENCES {column.type.table_name}({column.type.column_name})'
            return c + ", " + fk
        elif isinstance(column.type, BackRefType):
            return None
        elif isinstance(column.type, IdType):
            return f'{column.name} INTEGER{mod}'
        else:
            sql_type = {
                IntegerType(): 'INTEGER',
                StringType(): 'VARCHAR(4000)',
                # 'real': 'REAL',
            }[column.type]
            return f'{column.name} {sql_type}{mod}'

    def NamedTable(self, table: NamedTable):
        cols = [self.to_sql(r) for r in table.columns]
        return ' '.join(['CREATE TABLE', table.name, '(\n ', ',\n  '.join(
                c for c in cols if c is not None
            ), '\n);'])

    def AddRow(self, addrow: AddRow):
        cols = [c.name
                for c in self.state.tables[addrow.table].columns[1:]
                if not isinstance(c.type, BackRefType)]

        # values = [self._sqlexpr(v, {}) for v in addrow.args]
        # # TODO verify types
        # q = ['INSERT INTO', addrow.table.name, 
        #      "(", ', '.join(cols), ")",
        #      "VALUES",
        #      "(", ', '.join(v.sql for v in values), ")",
        # ]
        # insert = ' '.join(q) + ';'
        # return insert
        return ''

class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State()

        self.sqldecl = CompileSQL_Stmts(self.state)
        self.resolver = ResolveIdentifiers(self.state)

    def _add_table(self, table):
        self.state.tables[table.name] = table

        backrefs = []

        for c in table.columns:
            if c.backref:
                assert isinstance(c.type, RelationalType), c
                ref_to = self.state.tables[c.type.table_name]
                backrefs.append((ref_to, Column(c.backref, c.name, False, False, type=BackRefType(table.name))))

        for ref_to, col in backrefs:
            ref_to.columns.append(col)

        r = self.sqlengine.query( self.sqldecl.to_sql(table) )
        assert not r

    def _add_row(self, addrow: AddRow):
        insert = self.sqldecl.to_sql(addrow)
        assert not self.sqlengine.query(insert)

        if addrow.as_:
            rowid ,= self.sqlengine.query('SELECT last_insert_rowid();')[0]
            table = self.state.tables[addrow.table]
            idcol ,= [c for c in table.columns if c.name == 'id']
            compare = Compare('=', [idcol, Value(rowid, IntegerType())])
            self.state.vars[addrow.as_] = Projection(table, [compare] )

    def _def_function(self, func: Function):
        assert func.name not in self.state.functions
        self.state.functions[func.name] = func

    def run_stmt(self, c):
        if isinstance(c, NamedTable):
            self._add_table(c)
        elif isinstance(c, Function):
            self._def_function(c)
        elif isinstance(c, AddRow):
            self._add_row(c)
        else:
            raise ValueError(c)

    def call_func(self, fname, args):

        args = [Value.from_pyobj(a) for a in args]
        funccall = FuncCall(fname, FuncArgs(args, {}))
        # import pdb
        # pdb.set_trace()
        self.resolver.resolve(funccall)
        return funccall


def _test(fn):
    a = open(fn).read()
    # a = open("preql/simple2.pql").read()
    # a = open("preql/tree.pql").read()
    from .api import SqliteEngine
    sqlengine = SqliteEngine()
    i = Interpreter(sqlengine)
    for s in parse(a):
        i.run_stmt(s)

    # if isinstance(s, Function):
    #     print(s)
    #     print(list(resolver.resolve(s.expr)))
    #     print(s)
    #     print('-'*20)

    print('tables:', i.state.tables)
    print('vars:', i.state.vars)

    for name, f in i.state.functions.items():
        print('***', name, '***', f.params)
        print(f.expr)
        f = i.call_func(name, ["hello" for p in f.params or []])
        # print(f)
        print(f.to_tree().pretty())
        # expr = deepcopy(f.expr)
        # (i.resolver.resolve(expr))
        # print(expr)
        print('-'*20)

            
def test():
    # _test("preql/simple1.pql")
    _test("preql/simple2.pql")

if __name__ == '__main__':
    test()