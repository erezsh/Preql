from copy import deepcopy
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


class ExprCompiler:
    def __init__(self, interp):
        self.interp = interp
        self.functions = interp.functions
        self.vars = interp.vars
        self.tables = interp.tables
        self.context = Context()

    def _sqlexpr_RowRef(self, rowref: RowRef):
        return CompiledSQL(str(rowref.row_id), [IntType], ['id'])   # TODO id type

    def _sqlexpr(self, ast_node) -> CompiledSQL:
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr), (ast_node)
        res = getattr(self, '_sqlexpr_' + ast_type.__name__)(ast_node)
        assert isinstance(res, CompiledSQL), ast_node
        return res


    def _resolve_expr(self, ast_node):
        ast_type = type(ast_node)
        assert issubclass(ast_type, Expr), (ast_node)
        resolve_f = getattr(self, '_resolve_' + ast_type.__name__)
        if resolve_f is NotImplemented:
            return []
        return list(resolve_f(ast_node))

    def _sqlexpr_Value(self, v: Value):
        if v is Null:
            return CompiledSQL('NULL', [v.type])
        elif v.type is IntType:
            return CompiledSQL(str(v.value), [v.type])
        elif isinstance(v.type, ArrayType):
            values = [self._sqlexpr(v) for v in v.value]
            sql = 'VALUES(%s)' % ', '.join(_v.sql for _v in values)
            return CompiledSQL(sql, [v.type])
        assert v.type is StrType, v
        return CompiledSQL('"%s"' % v.value, [v.type])

    def _sqlexpr_ColumnRef(self, colref: ColumnRef):
        sql = '%s.%s' % (colref.alias or colref.relation.name, colref.column.name)
        col_prefix = '%s.' % colref.alias if colref.alias else '' 
        return CompiledSQL(sql, [colref.column.type], [col_prefix + colref.column.name])

    def _sqlexpr_TableType(self, tabletype: TableType):
        if tabletype.alias:
            sql = '%s %s' % (tabletype.name, tabletype.alias)
        else:
            sql = tabletype.name

        return CompiledSQL(sql, [tabletype], [tabletype.name])

    _resolve_Value = NotImplemented

    _resolve_ColumnRef = NotImplemented
    _resolve_TableType = NotImplemented
    _resolve_RowRef = NotImplemented

    def _resolve_exprs(self, exprs):
        for e in exprs:
            yield from self._resolve_expr(e)

    def _resolve_elems(self, expr: Expr):
        return self._resolve_exprs(expr.elems)
    _resolve_Arith = _resolve_elems
    _resolve_Compare = _resolve_elems

    def _resolve_Ref_relation(self, relation, alias, name, requires):
        c = relation.get_column(name)
        if isinstance(c.type, TableType):
            # Foreign Key - join with the target table
            requires.append( (c.type, None) )
        elif isinstance(c.type, BackRefType):
            # Many-to-one(?) - join with referencing table
            requires.append(( c.type.ref_to, relation.name ))
            return FuncCall('to_array', FuncArgs([c.name, c.type.ref_to], {}))

        return ColumnRef(relation, c, alias)

    def _resolve_Ref_base(self, name, requires):
        with suppress(KeyError):
            return self.context.get('args')[name]

        # Global variable?
        with suppress(KeyError):
            return self.vars[name]

        relation = self.context.get('relation')
        return self._resolve_Ref_relation(relation, None, name, requires)

    def _resolve_Ref(self, ref: Ref):
        assert ref.resolved is None, ref

        requires = []
        x = self._resolve_Ref_base(ref.name[0], requires)

        # Updates x as necessary (emulates recursion)
        for name in ref.name[1:]:
            if isinstance(x, RowRef):
                if name == 'id':
                    x = Value(IntType, x.row_id) 
                else:
                    relation = self.tables[x.relation.name]
                    c = relation.get_column(name)
                    if isinstance(c.type, TableType):
                        # assert False, (x, c, ref)
                        rowid ,= self.interp.sqlengine.query(f'select {c.name} from {relation.name} where id={x.row_id}')
                        x = RowRef(c.type, rowid[0])
                    else:
                        assert False, c
            elif isinstance(x, ColumnRef):
                relation = self.tables[x.column.type.name]
                # relation = self.tables[x.relation.name]
                c = relation.get_column(name)
                if isinstance(c.type, TableType):
                    yield c.type, None
                x = ColumnRef(relation, c)
            
            else:
                assert isinstance(x, TableType)
                x = self._resolve_Ref_relation(self.tables[x.name], x.alias, name, requires)
        
        ref.resolved = x
        yield from requires


    def _sqlexpr_Ref(self, ref: Ref):
        assert ref.resolved is not None, (ref, ref.resolved)
        return self._sqlexpr(ref.resolved)

    def _sqlexpr_Arith(self, arith: Arith):
        assert arith.op == '+'  # XXX temporary
        elems = [self._sqlexpr(e) for e in arith.elems]
        # TODO verify types match
        sql = ' UNION ALL '.join('(%s)'%e.sql for e in elems)
        return CompiledSQL(sql, elems[0].types)

    def _sqlexpr_Compare(self, compare: Compare):
        elems = [self._sqlexpr(e) for e in compare.elems]
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

    def _resolve_Join(self, join: Join):
        return self._resolve_expr(join.rel1) + self._resolve_expr(join.rel2)

    def _resolve_FreeJoin(self, join: FreeJoin):
        raise Exception()

    def _sqlexpr_Join(self, join: Join):
        r1 = self._sqlexpr(join.rel1)
        r2 = self._sqlexpr(join.rel2)
        assert len(r1.types) == len(r2.types) == 1  # TODO validate relation type

        # XXX temporary code, not generic enough!!
        # foreignkey ,= [c for c in t1.columns if c.type.name == join.rel2.name]
        # fkname = foreignkey.name
        fk, f, t = self._find_join(join.rel1, join.rel2)

        s = f'{r1.sql} JOIN {r2.sql} ON {f.name}.{fk.name} = {t.name}.id'
        return CompiledSQL(s, [RelationType])

    def _sqlexpr_FreeJoin(self, join: FreeJoin):
        r1 = self._sqlexpr(join.rel1)
        r2 = self._sqlexpr(join.rel2)

        s = f'{r1.sql} JOIN {r2.sql}'
        return CompiledSQL(s, [RelationType])

    def _resolve_relation(self, rel: Expr):
        if isinstance(rel, Join):
            self._resolve_relation(rel.rel1)
            self._resolve_relation(rel.rel2)
            return

        if isinstance(rel, TableType):
            # rel.resolved = rel
            # TODO what's here
            return

        assert isinstance(rel, Ref), rel
        assert not rel.resolved
        name ,= rel.name
        rel.resolved = TableType(name)

    def _resolve_Query(self, query: Query):
        if isinstance(query.relation, FuncCall):
            fc = query.relation
            assert fc.name == 'freejoin'
            assert not fc.args.pos_args
            relations = fc.args.named_args
            assert len(relations) == 2  # TODO support joining several at a time
            tables = {}
            for alias, ref in relations.items():
                assert isinstance(ref, Ref)
                name ,= ref.name
                tables[alias] = TableType(name, alias=alias)
            relation = FreeJoin(*tables.values(), None)
            args = dict(self.context.get('args'))
            args.update(tables)
            self.context.append( {'args': args} )
        else:
            table_name ,= query.relation.name   # Cannot handle join yet
            relation = TableType(table_name)
            self.context.append( {'relation': self.tables[table_name]} )
            tables = {}

        try:

            requires = []
            if query.selection:
                requires += self._resolve_exprs(query.selection)
            if query.projection:
                requires += self._resolve_exprs(query.projection)

            if requires:
                joined_tables = [t.name for t in tables.values()]
                for r, r_gb in requires:
                    if r.name in joined_tables:
                        continue
                    relation = Join(relation, r, [])
                    if r_gb:
                        assert query.groupby is None, query.groupby
                        query.groupby = r_gb
            
            self._resolve_relation(relation)
            query.relation = relation
        
        finally:
            self.context.pop()
        return []

    def _sqlexpr_Query(self, query: Query):
        table_name = query.relation.main_rel_name()

        if query.selection:
            # Verify allo types are BoolType
            sel = [self._sqlexpr(x) for x in query.selection]
            assert all(s.types==[BoolType] for s in sel)
            sel_sql = ' WHERE ' + ' AND '.join(s.sql for s in sel)
        else:
            sel_sql = ''

        if query.projection:
            proj = [self._sqlexpr(x) for x in query.projection]
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
            proj_types = [IntType]  # TODO id type
            proj_names = ['id']


        rel = self._sqlexpr(query.relation)
        assert len(rel.types) == 1  # TODO validate is a relation

        s = f'SELECT {proj_sql} FROM {rel.sql}{sel_sql}'
        if query.groupby:
            s += f' GROUP BY {query.groupby}.id'
        return CompiledSQL(s, proj_types, proj_names)

    def _resolve_FuncCall(self, funccall: FuncCall):
        assert not funccall.args.named_args # TODO

        if funccall.name in ('to_array', 'count'):
            yield from self._resolve_exprs(funccall.args.pos_args)
            return

        f = deepcopy(self.functions[funccall.name])
        funccall.resolved = f

        args_d = dict(zip(f.params or [], funccall.args.pos_args))

        assert funccall.name not in args_d
        self.context.append( {'args': args_d, 'func': f} )
        try:
            yield from self._resolve_expr(f.expr)
            yield from self._resolve_exprs(funccall.args.pos_args)
        finally:
            self.context.pop()

    def _sqlexpr_FuncCall(self, funccall: FuncCall):
        # TODO: Requires understanding type of arguments
        assert not funccall.args.named_args #TODO
        if funccall.name == 'to_array':
            name, expr = funccall.args.pos_args
            assert isinstance(expr, TableType)  # XXX temporary
            # TODO ArrayType(expr)
            return CompiledSQL('%s.id' % expr.name, [ArrayType(IntType)], [name])
        if funccall.name == 'count':
            args = [self._sqlexpr(a) for a in funccall.args.pos_args]
            # TODO validate all args are column names
            # TODO auto name field_count
            return CompiledSQL('count(%s)' % (', '.join(a.sql for a in args)), [IntType], ['count'])

        f = funccall.resolved #self.functions[funccall.name]
        assert f
        args = funccall.args
        assert len(args.pos_args) == len(f.params or []), (args, f.params)
        args_d = dict(zip(f.params or [], args.pos_args))
        assert funccall.name not in args_d
        self.context.append({'args': args_d, 'func': f})
        try:
            return self._sqlexpr(f.expr)
        finally:
            self.context.pop()

    def compile(self, expr):
        self._resolve_expr(expr)
        return self._sqlexpr(expr)



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

    def _sql_Column(self, column: Column):
        mod = ' NOT NULL' if not column.is_nullable else ''
        mod += ' PRIMARY KEY' if column.is_pk else ''

        if isinstance(column.type, TableType):
            c = f'{column.name} INTEGER{mod}'
            fk = f'FOREIGN KEY({column.name}) REFERENCES {column.type.name}(id)'
            return c + ", " + fk
        elif isinstance(column.type, BackRefType):
            return None
        else:
            sql_type = {
                IntType: 'INTEGER',
                StrType: 'VARCHAR(4000)',
                # 'real': 'REAL',
            }[column.type]
            return f'{column.name} {sql_type}{mod}'

    def _sql_Table(self, table: Table):
        cols = [self._sql(r) for r in table.columns]
        return ' '.join(['CREATE TABLE', table.name, '(\n ', ',\n  '.join(
                c for c in cols if c is not None
            ), '\n);'])

    def _add_table(self, table):
        self.tables[table.name] = table

        backrefs = []

        for c in table.columns:
            if c.backref:
                assert isinstance(c.type, TableType), c
                ref_to = self.tables[c.type.name]
                backrefs.append((ref_to, Column(c.backref, BackRefType(TableType(table.name)), c.name, False, False)))

        for ref_to, col in backrefs:
            ref_to.columns.append(col)

        r = self.sqlengine.query( self._sql(table) )
        assert not r

    def _sqlexpr(self, expr, _):
        return ExprCompiler(self).compile(expr)

    def _sql_AddRow(self, addrow: AddRow):
        cols = [c.name
                for c in self.tables[addrow.table.name].columns[1:]
                if not isinstance(c.type, BackRefType)]

        # requires = list(self._resolve_exprs(addrow.args, {}))
        # assert not requires, requires
        values = [self._sqlexpr(v, {}) for v in addrow.args]
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
        if fname in self.vars:
            assert not args
            sql = self._sqlexpr(self.vars[fname], {})
            return self._make_struct(sql.sql, sql)

        args = [Value.from_pyobj(a) for a in args]
        funccall = FuncCall(fname, FuncArgs(args, {}))
        # self._resolve_expr(funccall, {})
        funccall_sql = self._sqlexpr(funccall, {})
        return self._query_as_struct(funccall_sql)

    def _make_struct(self, row, compiled_sql):
        types = compiled_sql.types
        names = compiled_sql.names
        assert len(row) == len(names)
        assert len(row) == len(types)
        for i, n in enumerate(names):
            if n is None:
                names[i] = '_%d' % i
        assert len(names) == len(set(names)), names
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