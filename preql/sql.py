from .utils import dataclass, make_define_decorator

@dataclass
class CompiledSQL:
    text: str
    type: object


class Sql:
    pass


sqlclass = make_define_decorator(Sql)

@sqlclass
class Primitive(Sql):
    type: object
    text: str

    def compile(self):
        return CompiledSQL(self.text, self.type)

@sqlclass
class TableRef(Sql):
    type: object
    name: str
    # alias: str

    def compile(self):
        sql = self.name
        # if self.alias:
        #     sql += ' ' + self.alias
        return CompiledSQL(sql, self.type)

@sqlclass
class TableField(Sql):
    type: object
    alias: str
    columns: dict

    def compile(self):
        # return CompiledSQL(self.name, self.type)
        return CompiledSQL(', '.join(f'{self.alias}.{c}' for c in self.columns.keys()), self.type)

@sqlclass
class CountField(Sql):
    field: Sql
    type = int  # TODO correct object

    def compile(self):
        return CompiledSQL(f'count({self.field.compile().text})', self.type)

@sqlclass
class RoundField(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return CompiledSQL(f'round({self.field.compile().text})', self.type)


@sqlclass
class Compare(Sql):
    op: str
    exprs: [Sql]

    def __post_init__(self):
        for f in self.exprs or ():
            assert isinstance(f, Sql), f

    def compile(self):
        return CompiledSQL(self.op.join(e.compile().text for e in self.exprs), bool)    # TODO proper type

@sqlclass
class Arith(Sql):
    op: str
    exprs: [Sql]

    def compile(self):
        return CompiledSQL(self.op.join(e.compile().text for e in self.exprs), object)    # TODO derive proper type

@sqlclass
class Neg(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL("-" + s.text, s.type)

@sqlclass
class Desc(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL(s.text + " DESC", s.type)


@sqlclass
class ColumnRef(Sql):
    name: str
    # table_name: str = None

    def compile(self):
        s = self.name
        # if self.table_name:
        #     s = self.table_name + '.' + s
        return CompiledSQL(s, self)

@sqlclass
class ColumnAlias(Sql):
    name: str
    alias: str
    # table_name: str = None

    def compile(self):
        s = '%s AS %s' % (self.name, self.alias)
        # if self.table_name:
        #     s = self.table_name + '.' + s
        return CompiledSQL(s, self)


@sqlclass
class Select(Sql):
    type: object
    table: Sql
    fields: [Sql]
    conds: [Sql] = None
    group_by: [Sql] = None
    order: [Sql] = None
    offset: Sql = None
    limit: Sql = None

    def _init(self):
        assert self.fields, self

    def compile(self):
        fields_sql = [f.compile() for f in self.fields]
        select_sql = ', '.join(f.text for f in fields_sql)

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

@sqlclass
class Join(Sql):
    type: object
    tables: dict
    conds: [Sql]

    def compile(self):
        tables_sql = ['(%s)' % (t.compile().text) for name, t in self.tables.items()]
        join_sql = ' JOIN '.join(e for e in tables_sql)

        join_sql += ' ON ' + ' AND '.join(c.compile().text for c in self.conds)

        return CompiledSQL(join_sql, self.type)    # TODO joined type