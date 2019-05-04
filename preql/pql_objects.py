from .utils import dataclass, Dataclass, Context
from . import ast_classes as ast
from . import sql


class Object(Dataclass):
    pass

class Function(Object):
    pass

class Primitive(Object):
    value = None

    def repr(self, query_engine):
        return repr(self.value)

    def to_sql(self):
        return sql.Primitive(type(self), self.repr(None))

@dataclass
class Integer(Primitive):
    value: int

@dataclass
class Bool(Primitive):
    value: bool

@dataclass
class Float(Primitive):
    value: float

@dataclass
class String(Primitive):
    value: str

    def repr(self, sql_engine):
        return '"%s"' % self.value

@dataclass
class Row(Object):
    table: object
    attrs: dict

    def repr(self):
        attrs = ['%s: %r' % kv for kv in self.attrs.items()]
        return '{' + ', '.join(attrs) +'}'

    def __repr__(self):
        return self.repr()

@dataclass
class Table(Object):
    def getattr(self, name):
        if name in self.columns:
            return self.columns[name]

        if name == 'count':
            # TODO table should have a method dict
            return CountTable(self)
        elif name == 'limit':
            # TODO table should have a method dict
            return LimitTable(self)
        elif name == 'offset':
            # TODO table should have a method dict
            return OffsetTable(self)
        elif name == 'order':
            # TODO table should have a method dict
            return OrderTable(self)
        
        raise NameError(name)

    def _repr_table(self, query_engine, name):
        preview_limit = 3
        count = self.count(query_engine).value
        rows = self.query(query_engine, preview_limit)
        rows = [r.repr() for r in rows]
        if count > preview_limit:
            rows.append('... (%d more)' % (count - preview_limit))

        table_line = '<Table'
        if name:
            table_line += ':' + name
        table_line += ' count=' + str(count) + '>'
        lines = [table_line] + ['\t - %s' % (r,) for r in rows]
        return '\n'.join(lines)

    def count(self, query_engine):
        return query_engine.query( sql.Count(self.to_sql()) )

    def to_sql(self):
        raise NotImplementedError()

    def query(self, query_engine, limit=None):
        s = sql.Query(self, self.to_sql(), limit=limit)
        return query_engine.query(s)


    def cols_by_type(self, type_):
        return {name: c.col for name, c in self.columns.items()
                if isinstance(c.col.type, type_)}

    def from_sql_tuple(self, tup):
        proj = self.projection
        assert len(tup) == len(proj), (tup, proj)
        return Row(self, dict(zip(proj, tup)))

    def from_sql_tuples(self, tuples):
        return [self.from_sql_tuple(row) for row in tuples]

    @property
    def tuple_width(self):
        return len(self.projection)

    def repr(self, query_engine):
        return self._repr_table(query_engine, None)


@dataclass
class ColumnRef(Object):   # TODO proper hierarchy
    col: ast.Column
    table_alias: str = None

    def to_sql(self):
        return sql.ColumnRef(self.col.name, self.table_alias)


    @property
    def name(self):
        if self.table_alias:
           return '%s.%s' % (self.table_alias, self.col.name)
        return self.col.name

    @property
    def type(self):
        return self.col.type

@dataclass
class StoredTable(Table):
    tabledef: ast.TableDef

    def __post_init__(self):
        for c in self.tabledef.columns.values():
            assert isinstance(c, ast.Column), type(c)

        self.columns = {name:ColumnRef(c) 
                        for name, c in self.tabledef.columns.items()}

    def repr(self, query_engine):
        return self._repr_table(query_engine, self.name)

    def to_sql(self):
        return sql.TableRef(self, self.name)

    @property
    def name(self):
        return self.tabledef.name

    @property
    def projection(self):
        return {c.col.name:c.col.type for c in self.columns.values() 
                if not isinstance(c.col.type, ast.BackRefType)}

    @property
    def sql_namespace(self):
        return {c.col.name:c.col.type for c in self.columns.values() 
                if not isinstance(c.col.type, ast.BackRefType)}



@dataclass
class TableField(Table):
    "Table as a column"
    table: Table
    alias: str

    @property
    def projection(self):
        return self.table.projection

    @property
    def columns(self):
        return {name:ColumnRef(c.col, self.alias)
                for name, c in self.table.columns.items()}


    def to_sql(self):
        return sql.TableField(self.table, self.alias, self.table.columns)

    @property
    def type(self):
        return self.table
    @property
    def name(self):
        return self.alias


@dataclass
class TableMethod(Function):
    table: Table

@dataclass
class CountTable(TableMethod):

    def call(self, query_engine, args, named_args):
        assert not args, args
        assert not named_args
        return self.table.count(query_engine)

    def repr(self, query_engine):
        return f'<CountTable function>'

@dataclass
class LimitTable(TableMethod):
    def call(self, query_engine, args, named_args):
        assert not named_args
        limit ,= args
        return Query(self.table, limit=limit)

class OffsetTable(TableMethod):
    def call(self, query_engine, args, named_args):
        assert not named_args
        offset ,= args
        return Query(self.table, offset=offset)


@dataclass
class OrderTable(TableMethod):
    def call(self, query_engine, args, named_args):
        assert not named_args
        return Query(self.table, order=args)



@dataclass
class CountField(Function): # TODO not exactly function
    obj: ColumnRef
    type = Integer

    def to_sql(self):
        return sql.CountField( self.obj.to_sql() )
    
    @property
    def name(self):
        return f'count_{self.obj.name}'

@dataclass
class Round(Function):  # TODO not exactly function
    obj: Object

    def to_sql(self):
        return sql.Round( self.obj.to_sql() )

    @property
    def name(self):
        return f'round_{self.obj.name}'


@dataclass
class SqlFunction(Function):
    f: object

    def call(self, query_engine, args, named_args):
        return self.f(*args, **named_args)

@dataclass
class UserFunction(Function):
    funcdef: ast.FunctionDef

    # def call(self, query_engine, args):
    #     return self


@dataclass
class Query(Table):
    table: Table
    conds: list = None
    fields: list = None
    agg_fields: list = None
    order: list = None
    offset: Object = None
    limit: Object = None

    @property
    def name(self):
        raise NotImplementedError('Who dares ask the name of he who cannot be named?')

    def to_sql(self):
        # TODO assert types?
        fields = self.fields or [] #list(self.columns.values())
        agg_fields = self.agg_fields or []
        return sql.Query(
            type = self,
            table = self.table.to_sql(),
            conds = [c.to_sql() for c in self.conds or []],
            fields = [f.to_sql() for f in fields + agg_fields],
            group_by = [f.to_sql() for f in fields] if agg_fields else [],
            order = [o.to_sql() for o in self.order or []],
            offset = self.offset.to_sql() if self.offset else None,
            limit = self.limit.to_sql() if self.limit else None,
        )


    def repr(self, query_engine):
        return self._repr_table(query_engine, None)

    @property
    def projection(self):
        if self.fields:
            return {f.name:f.type for f in self.fields + self.agg_fields}
        return self.table.projection

    @property
    def columns(self):
        if self.fields:
            cols = {f.name: self.table.columns[f.expr.name]
                    for f in self.fields + self.agg_fields}

            return cols
        else:
            return self.table.columns

    @property   # TODO better mechanism?
    def tabledef(self):
        assert False
        t = self.table.tabledef
        if self.fields:
            cols = {f.name: t.columns[f.expr.name]
                    for f in self.fields + self.agg_fields}

            return ast.TableDef(None, cols)
        else:
            return t



@dataclass
class Compare(Object): # TODO Op? Function?
    op: str
    exprs: list

    def to_sql(self):
        return sql.Compare(self.op, [e.to_sql() for e in self.exprs])

@dataclass
class Arith(Object): # TODO Op? Function?
    op: str
    exprs: list

    def to_sql(self):
        return sql.Arith(self.op, [e.to_sql() for e in self.exprs])

@dataclass
class Neg(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        return sql.Neg(self.expr.to_sql())

@dataclass
class Desc(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        return sql.Desc(self.expr.to_sql())

@dataclass
class NamedExpr(Object):   # XXX this is bad but I'm lazy
    _name: str
    expr: Object

    def to_sql(self):
        return self.expr.to_sql()

    @property
    def name(self):
        return self._name or self.expr.name

    @property
    def type(self):
        return self.expr.type


@dataclass
class RowRef(Object):
    relation: Table
    row_id: int

    def to_sql(self):
        return sql.Primitive(Integer, str(self.row_id)) # XXX type = table?

@dataclass
class AutoJoin(Table):
    tables: dict

    def __init__(self, **tables):
        self._tables = tables

    @property
    def columns(self):
        return {n:TableField(t, n) for n,t in self._tables.items()}

    @property
    def projection_names(self):
        raise NotImplementedError()

    def to_sql(self):
        tables = self.columns
        assert len(tables) == 2

        ids =       [list(t.cols_by_type(ast.IdType).items())
                     for t in tables.values()]
        relations = [list(t.cols_by_type(ast.RelationalType).values())
                     for t in tables.values()]

        ids0, ids1 = ids
        id0 ,= ids0
        id1 ,= ids1
        name1, name2 = list(tables)
        
        assert len(ids) == 2
        assert len(relations) == 2
    
        table1 = id0[1].type.table
        table2 = id1[1].type.table

        to_join  = [(name1, c, name2) for c in relations[0] if c.type.table_name == table2]
        to_join += [(name2, c, name1) for c in relations[1] if c.type.table_name == table1]
        if len(to_join) > 1:
            raise Exception("More than 1 relation between %s <-> %s" % (table1.name, table2.name))
        to_join ,= to_join
        src_table, rel, dst_table = to_join

        tables = {name: t.table.to_sql() for name, t in tables.items()}
        conds = [sql.Compare('=', [sql.ColumnRef(rel.name, src_table), sql.ColumnRef('id', dst_table)])]
        return sql.Join(self, tables, conds)

        # conds += ' ON ' + f'{src_table}.{rel.name} = {dst_table}.id'   # rel.type.column_name

        # exprs_sql = ['(%s) %s' % (t.to_sql().text, name) for name, t in self.tables.items()]
        # join_sql = ' JOIN '.join(e for e in exprs_sql)

        # print(type(src_table), src_table)
        # join_sql += ' ON ' + f'{src_table}.{rel.name} = {dst_table}.id'   # rel.type.column_name

        # return CompiledSQL(join_sql, None)

    # def _find_relation(self, tables):
    #     resolved_tables = [t.resolved_table for t in tables]
    #     assert all(isinstance(t, NamedTable) for t in resolved_tables)
    #     table1, table2 = resolved_tables     # currently just 2 tables for now
    #     table1_name = table1.id.type.table
    #     table2_name = table2.id.type.table
    #     relations = [(table1, c, table2) for c in table1.relations if c.type.table_name == table2_name]
    #     relations += [(table2, c, table1) for c in table2.relations if c.type.table_name == table1_name]
    #     if len(relations) > 1:
    #         raise Exception("More than 1 relation between %s <-> %s" % (table1.name, table2.name))
    #     rel ,= relations
    #     src_table, rel, dst_table = rel
    #     return rel


    def from_sql_tuple(self, tup):
        items = {}
        for name, tbl in self.columns.items():
            subset = tup[:tbl.tuple_width]
            items[name] = tbl.from_sql_tuple(subset)

            tup = tup[tbl.tuple_width:]

        return Row(self, items)