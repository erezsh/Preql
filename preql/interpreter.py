from contextlib import suppress

from .ast_classes import *
from .parser import parse, parse_query

def find_exprs(expr):
    for e in expr.exprs:
        yield e
        yield from find_exprs(e)


@dataclass
class CompiledSQL:
    sql: str
    types: list
    names: list = None


class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.tables = {}
        self.functions = {}
        self.vars = {}

        self._rel_id = 0

    def _new_rel_name(self):
        n = 'r%d' % self._rel_id
        self._rel_id += 1
        return n

    def _sql(self, ast_node):
        ast_type = type(ast_node)
        assert not issubclass(ast_type, Expr)
        return getattr(self, '_sql_' + ast_type.__name__)(ast_node)

    def _sqlexpr(self, ast_node, context, requires):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr), (ast_node)
        res = getattr(self, '_sqlexpr_' + ast_type.__name__)(ast_node, context, requires)
        assert isinstance(res, CompiledSQL), ast_node
        return res

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

    def _sqlexpr_Value(self, v: Value, context, requires):
        if v.type is IntType:
            return CompiledSQL(str(v.value), [v.type])
        assert v.type is StrType, v
        return CompiledSQL('"%s"' % v.value, [v.type])


    def _bind_name(self, name, context, requires):
        # Function argument?
        with suppress(KeyError):
            return context['args'][name]

        # Global variable?
        with suppress(KeyError):
            return self.vars[name]

        return self._rel_attr(context['relation'], name, requires)

    def _rel_attr(self, relation, name, requires):
        c = relation.get_column(name)
        if isinstance(c.type, TableType):
            requires.append((c.type, None))
        elif isinstance(c.type, BackRefType):
            requires.append((c.type.ref_to, relation.name))
            # return 'array_agg(%s)' % c.type.ref_to.name
            return FuncCall('to_array', [c.name, '%s.id' % c.type.ref_to.name])
            # return ColumnRef(self.tables[c.type.ref_to.name], self.tables[c.type.ref_to.name].get_column('id'))
        return ColumnRef(relation, c)

    def _sqlexpr_ColumnRef(self, colref: ColumnRef, context, requires):
        sql = '%s.%s' % (colref.relation.name, colref.column.name)
        return CompiledSQL(sql, [colref.column.type], [colref.column.name])

    def _sqlexpr_TableType(self, tabletype: TableType, context, requires):
        return CompiledSQL(tabletype.name, [tabletype], [tabletype.name])

    def _sqlexpr_Ref(self, ref: Ref, context, requires):
        x = self._bind_name(ref.name[0], context, requires)
        for name in ref.name[1:]:
            if isinstance(x, RowRef):
                if name == 'id':
                    x = Value(IntType, x.row_id) 
                else:
                    relation = self.tables[x.relation.name]
                    c = relation.get_column(name)
                    if isinstance(c.type, TableType):
                        rowid ,= self.sqlengine.query(f'select id name from {relation.name} where id={x.row_id}')
                        x = RowRef(c.type, rowid[0])
                    else:
                        assert False, c
            else:
                assert isinstance(x, ColumnRef), x
                relation = self.tables[x.column.type.name]
                # relation = self.tables[x.relation.name]
                c = relation.get_column(name)
                if isinstance(c.type, TableType):
                    requires.append((c.type, None))
                x = ColumnRef(relation, c)
        return self._sqlexpr(x, {}, requires)


    def _sqlexpr_Compare(self, compare: Compare, context, requires):
        elems = [self._sqlexpr(e, context, requires) for e in compare.elems]
        # TODO verify types match, or convert if necessary
        sql = compare.op.join(e.sql for e in elems)
        return CompiledSQL(sql, [BoolType])

    def _find_join(self, rel1, rel2):
        t1 = self.tables[rel1.name]
        t2 = self.tables[rel2.name]
        for c in t1.columns:
            if isinstance(c.type, TableType) and c.type.name == rel2.name:
                return c, rel1, rel2
            elif isinstance(c.type, BackRefType) and c.type.ref_to.name == rel2.name:
                return t2.get_column(c.backref), rel2, rel1

    def _sqlexpr_Join(self, join: Join, context, requires):
        r1 = self._sqlexpr(join.rel1, context, requires)
        r2 = self._sqlexpr(join.rel2, context, requires)
        assert len(r1.types) == len(r2.types) == 1  # TODO validate relation type

        # XXX temporary code, not generic enough!!
        # foreignkey ,= [c for c in t1.columns if c.type.name == join.rel2.name]
        # fkname = foreignkey.name
        fk, f, t = self._find_join(join.rel1, join.rel2)

        s = f'{r1.sql} JOIN {r2.sql} ON {f.name}.{fk.name} = {t.name}.id'
        return CompiledSQL(s, [RelationType])


    def _sqlexpr_Query(self, query: Query, context, parent_requires):
        assert query.as_ is None, query
        assert isinstance(query.relation, Ref), query
        requires = []
        table_name ,= query.relation.name   # Cannot handle join yet
        context = dict(context)
        context['relation'] = self.tables[table_name]
        if query.selection:
            # Verify all types are BoolType
            sel = [self._sqlexpr(x, context, requires) for x in query.selection]
            assert all(s.types==[BoolType] for s in sel)
            sel_sql = ' WHERE ' + ' AND '.join(s.sql for s in sel)
        else:
            sel_sql = ''
        assert query.groupby is None
        if query.projection:
            proj = [self._sqlexpr(x, context, requires) for x in query.projection]
            proj_sql_list = []
            proj_types = []
            proj_names = []
            for p in proj:
                assert len(p.types) == 1
                assert p.names is None or len(p.names) == 1
                if len(p.types) == 1 and isinstance(p.types[0], ArrayType):
                    _sql = 'group_concat(%s)' % p.sql
                else:
                    _sql = p.sql
                proj_sql_list.append(_sql)
                proj_types += p.types
                proj_names += p.names or [None]*len(p.types) 

            proj_sql = ', '.join(proj_sql_list)
        else:
            proj_sql = table_name + '.id'
            # assert False, "Return struct"
            proj_types = [IntType]  # TODO id type
            proj_names = ['id']

        relation = TableType(table_name)
        groupby = None
        if requires:
            for r, r_gb in requires:
                relation = Join(relation, r, [])
                if r_gb:
                    assert groupby is None
                    groupby = r_gb

        rel = self._sqlexpr(relation, context, None)
        assert len(rel.types) == 1  # TODO validate is a relation

        s = f'SELECT {proj_sql} FROM {rel.sql}{sel_sql}'
        if groupby:
            s += f' GROUP BY {groupby}.id'
        return CompiledSQL(s, proj_types, proj_names)

        # selection: list
        # groupby: list
        # projection: list

    # def _sqlexpr_Function(self, func: Function, context):

    def _sqlexpr_FuncCall(self, funccall: FuncCall, context, requires):
        # TODO: Requires understanding type of arguments
        if funccall.name == 'to_array':
        #     assert False
        #     return '%s(%s)' % ('group_concat', ', '.join(funccall.args))
            name, expr = funccall.args
            # TODO always int???
            return CompiledSQL(expr, [ArrayType(IntType)], [name])
        if funccall.name == 'count':
            args = [self._sqlexpr(a, context, requires) for a in funccall.args]
            # TODO validate all args are column names
            # TODO auto name field_count
            return CompiledSQL('count(%s)' % (', '.join(a.sql for a in args)), [IntType], ['count'])

        f = self.functions[funccall.name]
        args = funccall.args
        assert len(args) == len(f.params or []), (args, f.params)
        args_d = dict(zip(f.params or [], args))
        return self._sqlexpr(f.expr, {'args': args_d}, requires)

    def _add_table(self, table):
        self.tables[table.name] = table

        for c in table.columns:
            if c.backref:
                assert isinstance(c.type, TableType)
                ref_to = self.tables[c.type.name]
                ref_to.columns.append( 
                    Column(c.backref, BackRefType(TableType(table.name)), c.name, False, False)
                )
        r = self.sqlengine.query( self._sql(table) )
        assert not r

    def _sqlexpr_RowRef(self, rowref: RowRef, context, requires):
        return CompiledSQL(str(rowref.row_id), [IntType], ['id'])   # TODO id type

    def _sql_AddRow(self, addrow: AddRow):
        cols = [c.name
                for c in self.tables[addrow.table.name].columns[1:]
                if not isinstance(c.type, BackRefType)]

        values = [self._sqlexpr(v, {}, None) for v in addrow.args]
        # TODO verify types
        q = ['INSERT INTO', addrow.table.name, 
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(v.sql for v in values), ")",
        ]
        insert = ' '.join(q) + ';'
        return insert

    def _add_row(self, addrow: AddRow):
        insert = self._sql(addrow)
        assert not self.sqlengine.query(insert)

        if addrow.as_:
            rowid ,= self.sqlengine.query('SELECT last_insert_rowid();')[0]
            self.vars[addrow.as_] = RowRef(addrow.table, rowid)

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
        funccall_sql = self._sqlexpr(funccall, {}, None)
        # return self.sqlengine.query(sql)
        return self._query_as_struct(funccall_sql)

    def _make_struct(self, row, compiled_sql):
        types = compiled_sql.types
        names = compiled_sql.names
        assert len(row) == len(names)
        assert len(row) == len(types)
        for i, n in enumerate(names):
            if n is None:
                names[i] = '_%d' % i
        assert len(names) == len(set(names))
        values = []
        for t, v in zip(types, row):
            if isinstance(t, ArrayType):
                assert t.elem_type is IntType   # XXX temporary
                values.append([int(e) for e in v.split(',')])
            else:
                values.append(v)

        return dict(zip(names, values))


    def _query_as_struct(self, compiled_sql):
        res = self.sqlengine.query(compiled_sql.sql)
        return [
            self._make_struct(row, compiled_sql)
            for row in res
        ]

    def execute(self, s):
        ast = parse(s)
        for stmt in ast:
            self.run_stmt(stmt)

    def query(self, q):
        ast = parse(q)
        sql = self.compile_query(ast)
        return self.sqlengine.query(sql)