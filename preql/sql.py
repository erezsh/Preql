from .utils import dataclass

@dataclass
class CompiledSQL:
    text: str
    type: object


class Sql:
    pass


@dataclass
class Primitive(Sql):
    type: object
    text: str

    def compile(self):
        return CompiledSQL(self.text, self.type)

@dataclass
class StoredTable(Sql):
    type: object
    name: str

    def compile(self):
        return CompiledSQL(self.name, self.type)

@dataclass
class CountField(Sql):
    field: Sql
    type = int  # TODO correct object

    def compile(self):
        return CompiledSQL(f'count({self.field.compile().text})', self.type)

@dataclass
class RoundField(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return CompiledSQL(f'round({self.field.compile().text})', self.type)


@dataclass
class Compare(Sql):
    op: str
    exprs: list

    def __post_init__(self):
        for f in self.exprs or ():
            assert isinstance(f, Sql), f

    def compile(self):
        return CompiledSQL(self.op.join(e.compile().text for e in self.exprs), bool)    # TODO proper type

@dataclass
class Arith(Sql):
    op: str
    exprs: list

    def compile(self):
        return CompiledSQL(self.op.join(e.compile().text for e in self.exprs), object)    # TODO derive proper type

@dataclass
class Neg(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL("-" + s.text, s.type)

@dataclass
class Desc(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL(s.text + " DESC", s.type)


@dataclass
class ColumnRef(Sql):
    name: str
    table_name: str = None

    def compile(self):
        s = self.name
        if self.table_name:
            s = self.table_name + '.' + s
        return CompiledSQL(s, self)


@dataclass
class Query(Sql):
    type: object
    table: Sql
    conds: list = None
    fields: list = None
    group_by: list = None
    order: list = None
    offset: Sql = None
    limit: Sql = None

    def __post_init__(self):
        for f in self.fields or ():
            assert isinstance(f, Sql), f
        for f in self.group_by or ():
            assert isinstance(f, Sql), f
        for f in self.conds or ():
            assert isinstance(f, Sql), f


    def compile(self):
        # assert self.fields
        if self.fields:
            fields_sql = [f.compile() for f in self.fields]
            select_sql = ', '.join(f.text for f in fields_sql)
        else:
            select_sql = '*'    # XXX Bad! Pass everything by name

        sql = f'SELECT {select_sql} FROM ({self.table.compile().text})'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile().text for c in self.conds)

        if self.group_by:
            sql += ' GROUP BY ' + ', '.join(e.compile().text for e in self.group_by)

        if self.limit:
            sql += ' LIMIT ' + self.limit.compile().text
        elif self.offset:
            sql += ' LIMIT -1'  # XXX Sqlite only

        if self.offset:
            sql += ' OFFSET ' + self.offset.compile().text

        if self.order:
            sql += ' ORDER BY ' + ', '.join(o.compile().text for o in self.order)

        return CompiledSQL(sql, self.type)

@dataclass
class Join(Sql):
    type: object
    tables: dict
    conds: list

    def compile(self):
        tables_sql = ['(%s) %s' % (t.compile().text, name) for name, t in self.tables.items()]
        join_sql = ' JOIN '.join(e for e in tables_sql)

        join_sql += ' ON ' + ' AND '.join(c.compile().text for c in self.conds)

        return CompiledSQL(join_sql, object)    # TODO joined type