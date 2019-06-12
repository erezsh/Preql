from .utils import dataclass, Dataclass, Context
from .utils import dataclass, make_define_decorator, field
from . import ast_classes as ast
from . import sql
from .exceptions import PreqlError_Attribute


class Object(Dataclass):
    tuple_width = 1

    def from_sql_tuple(self, tup):
        obj ,= tup
        return obj

pql_object = make_define_decorator(Object)


class Function(Object):
    pass

class Null(Object):
    def repr(self, query_engine):
        return 'null'

    def to_sql(self):
        return sql.null

null = Null()

class Primitive(Object):
    value = None

    def repr(self, query_engine):
        return repr(self.value)

    def to_sql(self):
        return sql.Primitive(type(self), self.repr(None))

@pql_object
class Integer(Primitive):
    value: int

@pql_object
class Bool(Primitive):
    value: bool

@pql_object
class Float(Primitive):
    value: float

@pql_object
class String(Primitive):
    value: str

    def repr(self, sql_engine):
        return '"%s"' % self.value

@pql_object
class Row(Object):
    table: object
    attrs: dict

    def repr(self):
        attrs = ['%s: %r' % kv for kv in self.attrs.items()]
        return '{' + ', '.join(attrs) +'}'

    def __repr__(self):
        return self.repr()



@pql_object
class Table(Object):
    def getattr(self, name):
        try:
            return self.get_column(name)
        except PreqlError_Attribute:
            pass

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
        
        raise PreqlError_Attribute(self.name, name)

    def get_column(self, name):
        try:
            col = self._columns[name]
        except KeyError:
            raise PreqlError_Attribute(self.name, name)

        col.invoked_by_user()
        return col

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
        return query_engine.query( sql.Select(Integer, self.to_sql(), fields=[sql.CountField(sql.Primitive(None, '*'))]) )

    def to_sql(self):
        raise NotImplementedError()

    def query(self, query_engine, limit=None):
        s = sql.Select(self, self.to_sql(), [sql.Primitive(None, '*')], limit=sql.Primitive(int, str(limit)) if limit!=None else None)
        return query_engine.query(s)


    def cols_by_type(self, type_):
        return {name: c.col for name, c in self._columns.items()
                if isinstance(c.col.type, type_)}

    def from_sql_tuple(self, tup):
        items = {}
        for name, col in self.projection.items():
            subset = tup[:col.tuple_width]
            items[name] = col.from_sql_tuple(subset)
            tup = tup[col.tuple_width:]

        assert not tup, (tup, self.projection)

        return Row(self, items)

    def from_sql_tuples(self, tuples):
        return [self.from_sql_tuple(row) for row in tuples]

    @property
    def projection(self):
        return {name:c for name, c in self._columns.items() 
                if not isinstance(c.type, ast.BackRefType) or c.invoked}

    @property
    def tuple_width(self):
        return len(self.projection)

    def repr(self, query_engine):
        return self._repr_table(query_engine, None)

@pql_object
class StoredTable(Table):
    tabledef: ast.TableDef
    query_state: object
    __columns = None

    def _init(self):
        for c in self.tabledef.columns.values():
            assert isinstance(c, ast.Column), type(c)

    @property
    def _columns(self): # Lazy to avoid self-recursion through columnref.relation
        if self.__columns is None:
            self.__columns = {name:ColumnRef(c, self) for name, c in self.tabledef.columns.items()}
        return self.__columns

    def repr(self, query_engine):
        return self._repr_table(query_engine, self.name)

    def to_sql(self, context=None):
        table_ref = sql.TableRef(self, self.name)
        columns = [sql.ColumnAlias(c.name, c.sql_alias) for c in self.projection.values()]
        return sql.Select(self, table_ref, columns)

    @property
    def name(self):
        return self.tabledef.name


MAKE_ALIAS = iter(range(1000))

@pql_object
class JoinableTable(Table):
    table: StoredTable
    joins: [StoredTable] = None

    def _init(self):
        object.__setattr__(self, 'joins', list(self.joins or ()))   # XXX ugly but best Python can do

    @property
    def query_state(self):
        return self.table.query_state

    @property
    def name(self):
        return self.table.name

    @property
    def _columns(self):
        return {name:ColumnRef(c.col, self, c.sql_alias) for name, c in self.table._columns.items()}

    def to_sql(self, context=None):
        if self.joins:
            x ,= self.joins
            if context and x in context['autojoined']:
                return self.table.to_sql()

            table = AutoJoin([self.table, x])
            if context is None:
                context = {'autojoined': [x]}
            else:
                context['autojoined'].append(x)
            return table.to_sql(context)

        return self.table.to_sql()



@pql_object
class ColumnRef(Object):   # TODO proper hierarchy
    col: ast.Column
    table: Table
    sql_alias: str = None
    relation: Table = None
    pql_name: str = None

    backref = None
    invoked = False

    def _init(self):
        if self.sql_alias is None:
            self._init_var('sql_alias', self.col.name + str(next(MAKE_ALIAS)))  # XXX ugly but that's the best Python has to offer
        if isinstance(self.type, ast.RelationalType):
            relation = self.table.query_state.get_table(self.type.table_name)
            self._init_var('relation', relation)

    def invoked_by_user(self):
        if isinstance(self.type, ast.BackRefType):
            if not self.backref:
                backref = self.table.query_state.get_table(self.col.table.name) # XXX kinda awkward
                if backref not in self.table.joins:
                    self.table.joins.append(backref)
                self.backref = backref
                self.invoked = True

    def to_sql(self):
        if isinstance(self.type, ast.BackRefType):
            assert self.backref
            return ( self.backref.get_column('id').to_sql() )

        return sql.ColumnRef(self.sql_alias)

    @property
    def name(self):
        return self.pql_name or self.col.name

    @property
    def type(self):
        return self.col.type

    def getattr(self, name):
        if isinstance(self.type, ast.BackRefType):
            assert self.backref in self.table.joins
            col = self.backref.get_column(name)
        else:
            assert isinstance(self.type, ast.RelationalType)
            if self.relation not in self.table.joins:
                self.table.joins.append(self.relation)
            col = self.relation.get_column(name)

        # TODO Store ColumnRef for correct re-use?
        new_col = ColumnRef(col.col, col.table, col.sql_alias, col.relation, self.name + '.' + col.name)
        new_col.invoked_by_user()
        return new_col

    def from_sql_tuple(self, tup):
        if isinstance(self.type, ast.RelationalType):
            # TODO return object instead of id
            pass

        return super().from_sql_tuple(tup)


@pql_object
class TableMethod(Function):
    table: Table

@pql_object
class CountTable(TableMethod):

    def call(self, query_engine, args, named_args):
        assert not args, args
        assert not named_args
        return self.table.count(query_engine)

    def repr(self, query_engine):
        return f'<CountTable function>'

@pql_object
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


@pql_object
class OrderTable(TableMethod):
    def call(self, query_engine, args, named_args):
        assert not named_args
        return Query(self.table, order=args)



@pql_object
class CountField(Function): # TODO not exactly function
    obj: ColumnRef
    type = Integer

    def to_sql(self):
        return sql.CountField( self.obj.to_sql() )
    
    @property
    def name(self):
        return f'count_{self.obj.name}'

@pql_object
class Round(Function):  # TODO not exactly function
    obj: Object
    type = Float

    def to_sql(self):
        return sql.RoundField( self.obj.to_sql() )

    @property
    def name(self):
        return f'round_{self.obj.name}'


@pql_object
class SqlFunction(Function):
    f: object

    def call(self, query_engine, args, named_args):
        return self.f(*args, **named_args)

@pql_object
class UserFunction(Function):
    funcdef: ast.FunctionDef

    # def call(self, query_engine, args):
    #     return self

@pql_object
class Query(Table):
    table: Table
    conds: list = None
    fields: list = None
    agg_fields: list = None
    order: list = None
    offset: Object = None
    limit: Object = None

    name = '<Query Object>'

    def _init(self):
        for f in self.fields or []:
            if isinstance(f.type, ast.BackRefType) or isinstance(f.expr, CountField): # XXX What happens if it's inside some expression??
                raise TypeError('Misplaced column "%s". Aggregated columns must appear after the aggregation operator "=>" ' % f.name)

    def to_sql(self, context=None):
        # TODO assert types?
        fields = self.fields or list(self.projection.values())
        agg_fields = self.agg_fields or []

        # Wrap array types with MakeArray
        agg_fields = [sql.MakeArray(f.to_sql()) if not isinstance(f.expr, CountField) else f.to_sql() for f in agg_fields]  # TODO be smarter about figuring that out

        fields = [f.to_sql() for f in fields]

        return sql.Select(
            type = self,
            table = self.table.to_sql(),
            fields = fields + agg_fields,
            conds = [c.to_sql() for c in self.conds or []],
            group_by = fields if agg_fields else [],
            order = [o.to_sql() for o in self.order or []],
            offset = self.offset.to_sql() if self.offset else None,
            limit = self.limit.to_sql() if self.limit else None,
        )

    def _to_sql(self):
        # TODO assert types?
        fields = self.fields or list(self.projection.values())
        agg_fields = self.agg_fields or []
        return sql.Select(
            type = self,
            table = self.table.to_sql(),
            fields = [f.to_sql() for f in fields + agg_fields],
            conds = [c.to_sql() for c in self.conds or []],
            group_by = [f.to_sql() for f in fields] if agg_fields else [],
            order = [o.to_sql() for o in self.order or []],
            offset = self.offset.to_sql() if self.offset else None,
            limit = self.limit.to_sql() if self.limit else None,
        )



    def repr(self, query_engine):
        return self._repr_table(query_engine, None)

    @property
    def _columns(self):
        if self.fields:
            return {f.name: f for f in self.fields + self.agg_fields}
        else:
            return self.table._columns



@pql_object
class Compare(Object): # TODO Op? Function?
    op: str
    exprs: list

    def to_sql(self):
        return sql.Compare(self.op, [e.to_sql() for e in self.exprs])

@pql_object
class Arith(Object): # TODO Op? Function?
    op: str
    exprs: list

    def to_sql(self):
        return sql.Arith(self.op, [e.to_sql() for e in self.exprs])

@pql_object
class Neg(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        return sql.Neg(self.expr.to_sql())

@pql_object
class Desc(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        return sql.Desc(self.expr.to_sql())

@pql_object
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

    @property
    def tuple_width(self):
        return self.expr.tuple_width

    def from_sql_tuple(self, tup):
        return self.expr.from_sql_tuple(tup)

    def invoked_by_user(self):
        return self.expr.invoked_by_user()

    @property
    def invoked(self):
        return self.expr.invoked


@pql_object
class RowRef(Object):
    relation: object
    row_id: int

    def to_sql(self):
        return sql.Primitive(Integer, str(self.row_id)) # XXX type = table?


@pql_object
class TableVariable(Table):
    name: str
    table: Table

    @property
    def _columns(self):
        return self.table._columns

    def to_sql(self):
        return self.table.to_sql()
    

@pql_object
class TableField(Table):
    table: TableVariable
    type = Table

    @property
    def _columns(self):
        return self.table._columns

    @property
    def name(self):
        return self.table.name

    def to_sql(self):
        return sql.TableField(self, self.name, self.projection)

    def invoked_by_user(self):
        pass

    def getattr(self, name):
        col = super().getattr(name)
        assert not isinstance(col.type, ast.BackRefType)
        # TODO Store ColumnRef for correct re-use?
        return ColumnRef(col.col, col.table, col.sql_alias, col.relation, self.name + '.' + col.name)


def create_autojoin(*args, **kwargs):
    tables = [TableVariable(k, v) for k, v in kwargs.items()]
    return AutoJoin(tables)

@pql_object
class AutoJoin(Table):
    tables: [Table]

    @property
    def _columns(self):
        return {t.name:TableField(t) for t in self.tables}

    def to_sql(self, context=None):
        tables = self.tables
        assert len(tables) == 2

        ids =       [list(t.cols_by_type(ast.IdType).items()) for t in tables]
        relations = [list(t.cols_by_type(ast.RelationalType).values())
                     for t in tables]

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

        if len(to_join) == 2 and list(reversed(to_join[1])) == list(to_join[0]):    # TODO XXX ugly hack!! This won't scale, figure out how to prevent this case
            to_join = to_join[:1]

        if len(to_join) > 1:
            raise Exception("More than 1 relation between %s <-> %s" % (table1, table2))

        to_join ,= to_join
        src_table, rel, dst_table = to_join

        tables = [src_table.to_sql(context), dst_table.to_sql(context)]
        key_col = src_table.get_column(rel.name).sql_alias
        dst_id = dst_table.get_column('id').sql_alias
        conds = [sql.Compare('=', [sql.ColumnRef(key_col), sql.ColumnRef(dst_id)])]
        return sql.Join(self, tables, conds)
