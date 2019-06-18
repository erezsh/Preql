from .utils import dataclass, make_define_decorator

class Sql:
    pass

@dataclass
class CompiledSQL:
    text: str
    type: Sql


sqlclass = make_define_decorator(Sql)

@sqlclass
class Null(Sql):
    def compile(self):
        return CompiledSQL('null', None)    # TODO null type

null = Null()

@sqlclass
class Atom(Sql):
    def import_value(self, value):
        return self.type(value[0][0])


@sqlclass
class Primitive(Atom):
    type: object
    text: str

    def compile(self):
        return CompiledSQL(self.text, self)




@sqlclass
class TableRef(Sql):
    type: object
    name: str

    def compile(self):
        return CompiledSQL(self.name, self)

@sqlclass
class TableField(Sql):
    type: object
    alias: str
    columns: dict

    def compile(self):
        return CompiledSQL(', '.join(f'{c.sql_alias}' for c in self.columns.values()), self)

@sqlclass
class CountField(Sql):
    field: Sql
    type = int  # TODO correct object

    def compile(self):
        return CompiledSQL(f'count({self.field.compile().text})', self)

@sqlclass
class LimitField(Sql):
    field: Sql
    limit: int
    # type = int  # TODO correct object

    def compile(self):
        return CompiledSQL(self.field.compile().text, self)
        
    def import_value(self, value):
        assert False

@sqlclass
class RoundField(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return CompiledSQL(f'round({self.field.compile().text})', self)

@sqlclass
class MakeArray(Sql):
    field: Sql
    type = list  # TODO correct object

    _sp = "\x01|\x02"

    def compile(self):
        # Sqlite Specific
        t = self.type
        return CompiledSQL(f'group_concat({self.field.compile().text}, "{self._sp}")', self)

    def import_value(self, value):
        return value.split(self._sp)

@sqlclass
class RoundField(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return CompiledSQL(f'round({self.field.compile().text})', self)


@sqlclass
class Compare(Sql):
    op: str
    exprs: [Sql]

    def __post_init__(self):
        for f in self.exprs or ():
            assert isinstance(f, Sql), f

    def compile(self):
        elems = [e.compile().text for e in self.exprs]
        compare = self.op.join(elems)
        return CompiledSQL(compare, bool)    # TODO proper type

@sqlclass
class Arith(Sql):
    op: str
    exprs: [Sql]

    def compile(self):
        return CompiledSQL(self.op.join(e.compile().text for e in self.exprs), self)    # TODO derive proper type

@sqlclass
class Neg(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL("-" + s.text, s)

@sqlclass
class Desc(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return CompiledSQL(s.text + " DESC", s)


@sqlclass
class ColumnRef(Sql):
    name: str

    def compile(self):
        return CompiledSQL(self.name, self)

@sqlclass
class ColumnRefs(Sql):
    "For grouping"

    refs: [ColumnRef]

    def compile(self):
        return CompiledSQL(', '.join([r.compile().text for r in self.refs]), self)

@sqlclass
class ColumnAlias(Sql):
    value: Sql
    alias: str

    def compile(self):
        if isinstance(self.value, ColumnRef) and self.value.name == self.alias:
            s = self.alias  # This is just for beauty, it's not necessary for function
        else:
            s = '%s AS %s' % (self.value.compile().text, self.alias)
        return CompiledSQL(s, self)

@sqlclass
class Insert(Sql):
    table: str
    cols: [str]
    values: [Sql]

    type = None

    def compile(self):

        q = ['INSERT INTO', self.table,
             "(", ', '.join(self.cols), ")",
             "VALUES",
             "(", ', '.join(v.compile().text for v in self.values), ")",
        ]
        insert = ' '.join(q) + ';'
        return CompiledSQL(insert, self)

    def import_value(self, value):
        assert not value
        return None

class LastRowId(Atom):
    type = int

    def compile(self):
        return CompiledSQL('SELECT last_insert_rowid()', self)


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

        return CompiledSQL(sql, self)

    def import_value(self, value):
        return self.type.from_sql_tuples(value)


@sqlclass
class Join(Sql):
    type: object
    tables: list
    conds: [Sql]

    def compile(self):
        tables_sql = ['(%s)' % (t.compile().text) for t in self.tables]
        join_sql = ' JOIN '.join(e for e in tables_sql)

        join_sql += ' ON ' + ' AND '.join(c.compile().text for c in self.conds)

        return CompiledSQL(join_sql, self)    # TODO joined type