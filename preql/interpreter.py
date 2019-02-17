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
        raise KeyError(name, d.keys())

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


class CompileSQL_Expr:
    def __init__(self, state, mention_table: bool, required_rows: list):
        self.state = state
        # self.context = Context()
        self.mention_table = mention_table
        self.required_rows = required_rows

    def compile(self, expr):
        sql = self._sqlexpr(expr)
        if self.required_rows:
            with_parts = [ f'{r.autoname} AS (SELECT * FROM {r.table.name} WHERE id={r.row_id})'
                          for r in self.required_rows]
            with_sql = 'WITH ' + ', '.join(with_parts)
            sql.sql = with_sql + sql.sql

        return sql

    def _sqlexpr(self, ast_node) -> CompiledSQL:
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr), (ast_node)
        res = getattr(self, ast_type.__name__)(ast_node)
        assert isinstance(res, CompiledSQL), ast_node
        return res


    def _resolved(self, node):
        assert node.resolved
        return self._sqlexpr(node.resolved)

    FuncCall = _resolved
    Identifier = _resolved

    def Query(self, query: Query):
        table_sql = self._sqlexpr(query.table)

        where_sql = None
        if query.selection:
            exprs_sql = [self._sqlexpr(e) for e in query.selection]
            where_sql = ' AND '.join(e.sql for e in exprs_sql)

        if query.projection:
            exprs_sql = [self._sqlexpr(e) for e in query.projection + query.aggregates]
            proj_sql = ', '.join(e.sql for e in exprs_sql)
            proj_types = [e.types[0] for e in exprs_sql]
        else:
            proj_sql = '*'
            proj_types = table_sql.types
        
        # TODO groupby changes projection types
        if query.aggregates:
            exprs_sql = [self._sqlexpr(e) for e in query.projection]
            groupby_sql = ', '.join(e.sql for e in exprs_sql)
        else:
            groupby_sql = None

        if query.order:
            exprs_sql = [self._sqlexpr(e) for e in query.order]
            order_sql = ', '.join(e.sql for e in exprs_sql)
        else:
            order_sql = None




        sql = f'SELECT {proj_sql} FROM ({table_sql.sql})'
        if where_sql:
            sql += ' WHERE ' + where_sql
        if order_sql:
            sql += ' ORDER BY ' + order_sql
        if groupby_sql:
            sql += ' GROUP BY ' + groupby_sql
        return CompiledSQL(sql, proj_types, [])

    # def Projection(self, proj):
    #     # TODO names
    #     exprs_sql = [self._sqlexpr(e) for e in proj.exprs]
    #     assert all(len(e.types) == 1 for e in exprs_sql)
    #     # assert all(len(e.names) == 1 for e in exprs_sql), exprs_sql
    #     # proj_names = [e.names[0] for e in exprs_sql]
    #     sql = f'SELECT {proj_sql} FROM ({table_sql.sql})'
    #     return CompiledSQL(sql, proj_types, [])

    # def Selection(self, sel):
    #     table_sql = self._sqlexpr(sel.table)
    #     exprs_sql = [self._sqlexpr(e) for e in sel.exprs]
    #     sql = f'SELECT * FROM ({table_sql.sql}) WHERE ({where_sql})'
    #     return CompiledSQL(sql, table_sql.types, table_sql.names)

    def NamedTable(self, nt):
        # XXX returns columns? Just IdType?
        return CompiledSQL(nt.name, [nt], [nt.name])

    def AliasedTable(self, at):
        sql = self._sqlexpr(at.table)
        return CompiledSQL(f'({sql.sql}) {at.name}', sql.types, sql.names)

    def Compare(self, cmp):
        exprs_sql = [self._sqlexpr(e) for e in cmp.exprs]
        assert all(len(e.types) == 1 for e in exprs_sql)
        types = [e.types[0] for e in exprs_sql]
        # assert all(t==types[0] for t in types[1:]), types

        return CompiledSQL(cmp.op.join(e.sql for e in exprs_sql), [cmp.type])

    def Column(self, col):
        # TODO always create aliased table on joins, and remove mention_table?
        prefix = ''
        if isinstance(col.table, AliasedTable) or self.mention_table:
            prefix = col.table.name + '.'
        return CompiledSQL(prefix + col.name, [col.type], [col.name])

    def Value(self, v):
        # if v is Null:
        #     return CompiledSQL('NULL', [v.type])
        # elif v.type is IntType:
        #     return CompiledSQL(str(v.value), [v.type])
        # elif isinstance(v.type, ArrayType):
        #     values = [self._sqlexpr(v) for v in v.value]
        #     sql = 'VALUES(%s)' % ', '.join(_v.sql for _v in values)
        #     return CompiledSQL(sql, [v.type])
        # assert v.type is StrType, v
        if isinstance(v.type, StringType):
            return CompiledSQL('"%s"' % v.value, [v.type])
        elif isinstance(v.type, IntegerType):
            return CompiledSQL('%s' % v.value, [v.type])
        assert False

    def NamedExpr(self, ne):
        expr_sql = self._sqlexpr(ne.expr)
        # TODO add name here
        return CompiledSQL(expr_sql.sql, expr_sql.types, expr_sql.names)

    def Count(self, cnt):
        if cnt.exprs:
            exprs_sql = [self._sqlexpr(e) for e in cnt.exprs]
            count_sql = ', '.join(e.sql for e in exprs_sql)
        else:
            count_sql = '*'
        return CompiledSQL(f'count({count_sql})', [cnt.type])

    def _find_relation(self, tables):
        resolved_tables = [t.resolved_table for t in tables]
        assert all(isinstance(t, NamedTable) for t in resolved_tables)
        table1, table2 = resolved_tables     # currently just 2 tables for now
        table1_name = table1.id.type.table
        table2_name = table2.id.type.table
        relations = [(table1, c, table2) for c in table1.relations if c.type.table_name == table2_name]
        relations += [(table2, c, table1) for c in table2.relations if c.type.table_name == table1_name]
        if len(relations) > 1:
            raise Exception("More than 1 relation between %s <-> %s" % (table1.name, table2.name))
        rel ,= relations
        return rel


    def AutoJoin(self, autojoin):
        src_table, rel, dst_table = self._find_relation(autojoin.exprs)
        exprs_sql = [self._sqlexpr(e) for e in autojoin.exprs]
        join_sql = ' JOIN '.join(e.sql for e in exprs_sql)

        join_sql += ' ON ' + f'{src_table.name}.{rel.name} = {dst_table.name}.id'   # rel.type.column_name
        return CompiledSQL(join_sql, sum([e.types for e in exprs_sql], []))

    def FreeJoin(self, freejoin):
        exprs_sql = [self._sqlexpr(e) for e in freejoin.exprs]
        join_sql = ' JOIN '.join(e.sql for e in exprs_sql)
        return CompiledSQL(join_sql, sum([e.types for e in exprs_sql], []))

    def RowRef(self, rowref):
        # TODO: assert type?
        return CompiledSQL(str(rowref.row_id), [rowref.type])

    def ValueRef(self, valueref):
        # TODO get actual type
        sql = f'(SELECT {valueref.column} FROM {valueref.rowref.autoname})'
        return CompiledSQL(sql, [None])

    def OrderSpecifier(self, order_spec):
        expr_sql = self._sqlexpr(order_spec.expr)
        spec_sql = expr_sql.sql + ' ' + ('ASC' if order_spec.asc else 'DESC')
        return CompiledSQL(spec_sql, expr_sql.types, expr_sql.names)

    def Limit(self, lim):
        table_sql = self._sqlexpr(lim.args['tab'])
        limit_sql = self._sqlexpr(lim.args['maxlen'])
        sql = table_sql.sql + ' LIMIT ' + limit_sql.sql
        return CompiledSQL(sql, table_sql.types, table_sql.names)

    def Offset(self, offset):
        table_sql = self._sqlexpr(offset.args['tab'])
        start_sql = self._sqlexpr(offset.args['start'])
        sql = table_sql.sql + ' OFFSET ' + start_sql.sql
        return CompiledSQL(sql, table_sql.types, table_sql.names)

class ResolveIdentifiers:

    def __init__(self, state):
        self.state = state
        # self.functions = interp.functions
        # self.vars = interp.vars
        # self.tables = interp.tables
        self.context = Context()
        self.autojoins = []
        self.autojoined = False
        self.required_rows = []

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

    def Query(self, query: Query):
        assert not self.autojoined, "Not implemented yet"

        self._resolve_expr(query.table)

        assert not self.autojoins   # should be cleared by this point


        query.resolved_table = query.table.resolved_table
        self.context.append({'table': query.resolved_table})
        self._resolve_expr_list(query.selection)
        self._resolve_expr_list(query.projection)
        self._resolve_expr_list(query.aggregates)

        # if query.aggregates:
        #     import pdb
        #     pdb.set_trace()

        # self._resolve_expr_list(query.groupby or [])   # XXX not needed because it shares with projection. Ugly?
        self._resolve_expr_list(query.order)
        self.context.pop()

        # TODO: Verify there are no freejoins. Are explicit autojoins allowed?
        for needed_table in self.autojoins:
            assert needed_table != query.table
            query.table = AutoJoin([query.table, needed_table])

        self.autojoins = []
        self.autojoined = True


    def OrderSpecifier(self, order_spec):
        self._resolve_expr(order_spec.expr)

    # def Projection(self, proj: Projection):
    #     assert proj.resolved_table, proj
    #     self._resolve_exprs(proj)
    #     self.context.pop()
    #     # TODO return projected table

    # def Selection(self, sel: Selection):
    #     sel.resolved_table = sel.table.resolved_table
    #     assert sel.resolved_table, sel
    #     self.context.append({'table': sel.resolved_table})
    #     self._resolve_exprs(sel)
    #     self.context.pop()

    def Identifier(self, ident: Identifier):
        assert not ident.resolved, ident

        basename = ident.name[0]
        # assert not ident.name[1:]   # TODO

        args = self.context.get('args')

        path = ident.name[1:]
        if basename in args:
            obj = args[basename]
        elif basename in self.state.vars:
            obj = self.state.vars[basename]

        elif basename in self.state.tables:
            obj = self.state.tables[basename]
        else:
            # Check if basename is a column in the current table
            try:
                obj = self.context.get('table')
            except KeyError:
                raise Exception("Failed to resolve identifier", ident, args)

            path = ident.name
            assert obj

        if path:
            assert isinstance(obj.type, TabularType), obj

            for name in path:
                obj = obj.resolved_table[name]

                if isinstance(obj.type, RelationalType):
                    # assert False
                    obj = self.state.tables[obj.type.table_name]
                    if obj not in self.autojoins:
                        self.autojoins.append(obj)
                # elif isinstance(obj.type, TabularType):
                #     assert False
                elif isinstance(obj, ValueRef):
                    if obj.rowref not in self.required_rows:
                        self.required_rows.append(obj.rowref)
                else:
                    assert isinstance(obj, (Column, Value))
                    if isinstance(obj.type, BackRefType):
                        obj = self.state.tables[obj.type.to_table]
                        if obj not in self.autojoins:
                            self.autojoins.append(obj)
                        obj = obj.resolved_table['id']  # XXX hack, shouldn't work like this
                    else:
                        assert isinstance(obj.type, AtomType), obj


        if isinstance(obj, AliasedTable):
            obj = obj.resolved_table['id']     # XXX a bit of a hack to support freejoins

        ident.resolved = obj
        return ident.resolved

    def NamedExpr(self, named_expr):
        self._resolve_expr(named_expr.expr)

    def AutoJoin(self, join: AutoJoin):
        # TODO: Add join condition
        self._resolve_expr_list(join.exprs)
        join.resolved_table = join

    def FreeJoin(self, join: FreeJoin):
        self._resolve_expr_list(join.exprs)
        join.resolved_table = join

    def AliasedTable(self, aliased):
        # TODO clunky code. Put into generic methods
        self._resolve_expr(aliased.table)
        resolved = aliased.table.resolved_table
        columns = [copy(c) for c in resolved.columns]
        renamed_table = NamedTable(columns, aliased.name)
        for c in columns:
            c.table = aliased #renamed_table
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

    def Limit(self, lim):
        lim.args = self.context.get('args')
        # TODO ???

    def Offset(self, offset):
        offset.args = self.context.get('args')
        # TODO ???


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


        resolver = ResolveIdentifiers(self.state)
        resolver.context.append({'args': []})
        for v in addrow.args:
            resolver.resolve(v)
        expr_compiler = CompileSQL_Expr(self.state, resolver.autojoined, resolver.required_rows)
        values = [expr_compiler.compile(v) for v in addrow.args]
        # TODO verify types
        q = ['INSERT INTO', addrow.table,
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(v.sql for v in values), ")",
        ]
        insert = ' '.join(q) + ';'
        return insert



class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State()
        self.state.functions['limit'] = BuiltinFunction('limit', ['tab', 'maxlen'], Limit())
        self.state.functions['offset'] = BuiltinFunction('offset', ['tab', 'start'], Offset())

        self.sqldecl = CompileSQL_Stmts(self.state)

    def _add_table(self, table):
        self.state.tables[table.name] = table

        backrefs = []

        for c in table.columns:
            if c.backref:
                assert isinstance(c.type, RelationalType), c
                ref_to = self.state.tables[c.type.table_name]
                backrefs.append((ref_to, Column(c.backref, c.name, False, False, type=BackRefType(table.name), table=table)))

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
            # idcol ,= [c for c in table.columns if c.name == 'id']
            # compare = Compare('=', [idcol, Value(rowid, IntegerType())])
            v = RowRef(table, rowid)
            self.state.vars[addrow.as_] = v # Projection(table, [compare] )

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

    def _compile_func(self, fname, args):

        args = [Value.from_pyobj(a) for a in args]
        funccall = FuncCall(fname, FuncArgs(args, {}))

        resolver = ResolveIdentifiers(self.state)
        resolver.resolve(funccall)
        expr_compiler = CompileSQL_Expr(self.state, resolver.autojoined, resolver.required_rows)

        return expr_compiler.compile(funccall)
        # return self._query_as_struct(funccall_sql)

    def call_func(self, fname, args):
        sql = self._compile_func(fname, args)
        assert isinstance(sql, CompiledSQL)
        return self._query_as_struct(sql)

    def _query_as_struct(self, compiled_sql):
        res = self.sqlengine.query(compiled_sql.sql)
        return res
        # return [ self._make_struct(row, compiled_sql) for row in res ]

    def execute_code(self, code):
        for s in parse(code):
            self.run_stmt(s)



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
        if isinstance(f, BuiltinFunction) or f.name.startswith('_'):
            continue

        print('***', name, '***', f.params)
        print(f.expr)
        f = i.call_func(name, ["hello" for p in f.params or []])
        print(f)
        # print(f.sql)
        # print(f.to_tree().pretty())
        # expr = deepcopy(f.expr)
        # (i.resolver.resolve(expr))
        # print(expr)
        print('-'*20)


def test():
    # _test("preql/simple1.pql")
    _test("preql/simple2.pql")
    # _test("preql/tree.pql")

if __name__ == '__main__':
    test()