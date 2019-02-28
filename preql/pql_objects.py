from .utils import dataclass, Dataclass, Context
from . import ast_classes as ast

@dataclass
class CompiledSQL:
    text: str
    type: object


class Object(Dataclass):
    pass

class Function(Object):
    pass

class Primitive(Object):
    value = None

    def repr(self, query_engine):
        return repr(self.value)

    def to_sql(self):
        return CompiledSQL(self.repr(None), self)

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
    attrs: dict

    def repr(self):
        attrs = ['%s: %r' % kv for kv in self.attrs.items()]
        return '{' + ', '.join(attrs) +'}'

@dataclass
class Table(Object):
    def getattr(self, name):
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
        
        try:
            col = self.tabledef.columns[name]
        except KeyError:
            raise NameError(name)

        return ColumnRef(col)

        # raise AttributeError(name)

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
        # TODO Optimization: remove query order when running count
        sql = self.to_sql()
        sql = CompiledSQL(f'SELECT count(*) from ({sql.text})', Integer)
        return query_engine.query(sql)

    def to_sql(self):
        raise NotImplementedError()

    def query(self, query_engine, limit=None):
        sql_text = self.to_sql().text
        if limit is not None:   # TODO use Query.limit, not this hack
            sql_text = 'SELECT * FROM (' + sql_text + ') LIMIT %d' % limit
        sql = CompiledSQL(sql_text, ObjectFromTuples(self))
        return query_engine.query(sql)

@dataclass
class ObjectFromTuples(Dataclass):   # TODO what is this
    table: Table

    def __call__(self, rows):
        names = self.table.projection_names
        x = []
        for row in rows:
            assert len(row) == len(names), (row, names)
            x.append( Row(dict(zip(names, row))) )
        return x




@dataclass
class ColumnRef(Object):   # TODO proper hierarchy
    col: ast.Column

    def to_sql(self):
        # TODO use table name
        return CompiledSQL(self.col.name, self)


    @property
    def name(self):
        return self.col.name

@dataclass
class StoredTable(Table):
    tabledef: ast.TableDef

    def funcname(self, parameter_list):
        pass

    def repr(self, query_engine):
        # count = self.count(query_engine)
        # return f'<Table:{self.name} count={count.value}>'
        return self._repr_table(query_engine, self.name)

    def to_sql(self):
        return CompiledSQL(self.name, self)

    @property
    def name(self):
        return self.tabledef.name

    @property
    def projection_names(self):
        return [c.name for c in self.tabledef.columns.values() if not isinstance(c.type, ast.BackRefType)]

@dataclass
class TableMethod(Function):
    table: Table

@dataclass
class CountTable(TableMethod):

    def call(self, query_engine, args):
        assert not args, args
        return self.table.count(query_engine)

    def repr(self, query_engine):
        return f'<CountTable function>'

@dataclass
class LimitTable(TableMethod):
    def call(self, query_engine, args):
        limit ,= args
        return Query(self.table, limit=limit)

class OffsetTable(TableMethod):
    def call(self, query_engine, args):
        offset ,= args
        return Query(self.table, offset=offset)


@dataclass
class OrderTable(TableMethod):
    def call(self, query_engine, args):
        return Query(self.table, order=args)



@dataclass
class CountField(Function): # TODO not exactly function
    obj: ColumnRef

    def to_sql(self):
        obj_sql = self.obj.to_sql()
        return CompiledSQL(f'count({obj_sql.text})', Integer)
    
    @property
    def name(self):
        return f'count_{self.obj.name}'

@dataclass
class Round(Function):  # TODO not exactly function
    obj: Object

    def to_sql(self):
        obj_sql = self.obj.to_sql()
        return CompiledSQL(f'round({obj_sql.text})', Integer)

    @property
    def name(self):
        return f'round_{self.obj.name}'


@dataclass
class SqlFunction(Function):
    f: object

    def call(self, query_engine, args):
        return self.f(*args)

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

    # def __init__(self, table, conds=None, fields=None, agg_fields=None,
    #                           order=None, offset=None, limit=None):
    #     self.table = table
    #     self.conds = conds
    #     self.fields = fields
    #     self.agg_fields = agg_fields
    #     self.order = order
    #     self.offset = offset
    #     self.limit = limit

    def to_sql(self):
        # TODO assert all boolean?
        table_sql = self.table.to_sql()

        if self.conds:
            conds_sql = [c.to_sql() for c in self.conds]
            where_sql = ' AND '.join(c.text for c in conds_sql)
        else:
            where_sql = ''

        if self.fields:
            fields_sql = [f.to_sql() for f in self.fields + self.agg_fields]
            # TODO assert all boolean?
            select_sql = ', '.join(f.text for f in fields_sql)
        else:
            select_sql = '*'

        if self.agg_fields:
            agg_sql = [f.to_sql() for f in self.fields]
            groupby_sql = ', '.join(e.text for e in agg_sql)
        else:
            groupby_sql = None

        sql = f'SELECT {select_sql} FROM ({table_sql.text})'
        if where_sql:
            sql += ' WHERE ' + where_sql
        # if order_sql:
        #     sql += ' ORDER BY ' + order_sql
        if groupby_sql:
            sql += ' GROUP BY ' + groupby_sql

        if self.limit:
            sql += ' LIMIT ' + self.limit.to_sql().text
        elif self.offset:
            sql += ' LIMIT -1'  # XXX Sqlite only

        if self.offset:
            sql += ' OFFSET ' + self.offset.to_sql().text


        if self.order:
            order_exprs_sql = [o.to_sql() for o in self.order]
            order_sql = ', '.join(o.text for o in order_exprs_sql)
            sql += ' ORDER BY ' +  order_sql

        return CompiledSQL(sql, self.table)

    def repr(self, query_engine):
        return self._repr_table(query_engine, None)

    @property
    def projection_names(self):
        if self.fields:
            return [f.name for f in self.fields + self.agg_fields]
        return self.table.projection_names

    @property   # TODO better mechanism?
    def tabledef(self):
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
        return CompiledSQL(self.op.join(e.to_sql().text for e in self.exprs), self)

@dataclass
class Arith(Object): # TODO Op? Function?
    op: str
    exprs: list

    def to_sql(self):
        return CompiledSQL(self.op.join(e.to_sql().text for e in self.exprs), self)

@dataclass
class Neg(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        sql = self.expr.to_sql()
        return CompiledSQL("-" + sql.text, sql.type)

@dataclass
class Desc(Object): # TODO Op? Function?
    expr: Object

    def to_sql(self):
        sql = self.expr.to_sql()
        return CompiledSQL(sql.text + " DESC", sql.type)

@dataclass
class NamedExpr(Object):   # XXX this is bad but I'm lazy
    _name: str
    expr: Object

    def to_sql(self):
        return self.expr.to_sql()

    @property
    def name(self):
        return self._name or self.expr.name
