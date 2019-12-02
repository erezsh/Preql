from typing import List, Any, Optional

from .utils import dataclass, SafeDict, listgen
from . import pql_types as types

@dataclass
class Sql:
    type: types.PqlType

    def _compile(self, sql_code):
        return CompiledSQL(sql_code, self)

    # def __created__(self):
    #     for name in self.__annotations__:
    #         var = getattr(self, name)
    #         if isinstance(var, Sql):
    #             print("@ ", var)

@dataclass
class CompiledSQL:
    text: str
    sql: Sql

@dataclass
class RawSql(Sql):
    text: str

    def compile(self):
        return self._compile(self.text)


@dataclass
class Null(Sql):
    def compile(self):
        return self._compile('null')    # TODO null type

null = Null(types.null)

@dataclass
class Atom(Sql):

    def import_result(self, res):
        row ,= res
        item ,= row
        return item


@dataclass
class Primitive(Atom):
    text: str

    def compile(self):
        return self._compile(self.text)


class Table(Sql):

    @listgen
    def import_result(self, arr):
        expected_length = self.type.flat_length()
        for row in arr:
            assert len(row) == expected_length
            i = iter(row)
            s = ({str(name): col.type.restructure_result(i) for name, col in self.type.columns.items()})
            yield s

@dataclass
class TableName(Table):
    name: str

    def compile(self):
        return self._compile(self.name)

@dataclass
class Count(Sql):
    field: Sql
    type = types.Int  # TODO correct object

    def compile(self):
        return self._compile(f'count({self.field.compile().text})')

@dataclass
class Round(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return self._compile(f'round({self.field.compile().text})')

@dataclass
class MakeArray(Sql):
    field: Sql
    type = list  # TODO correct object

    _sp = "|"

    def compile(self):
        # Sqlite Specific
        # t = self.type
        return self._compile(f'group_concat({self.field.compile().text}, "{self._sp}")')

    def import_result(self, value):
        assert False
        if value is None:
            return []
        return value.split(self._sp)


@dataclass
class Contains(Sql):
    op: str
    exprs: List[Sql]

    def compile(self):
        assert self.op
        elems = ['(%s)' % e.compile().text for e in self.exprs]
        contains = (' %s ' % self.op).join(elems)
        return self._compile(contains)


@dataclass
class Compare(Sql):
    op: str
    exprs: List[Sql]

    def __post_init__(self):
        for f in self.exprs or ():
            assert isinstance(f, Sql), f

    def compile(self):
        elems = [e.compile().text for e in self.exprs]
        compare = (' %s ' % self.op).join(elems)
        return self._compile(compare)

@dataclass
class Arith(Sql):
    op: str
    exprs: List[Sql]

    def compile(self):
        x = (' %s ' % self.op).join(e.compile().text for e in self.exprs)
        return self._compile('(%s)'%x)


class TableArith(Arith, Table):
    pass

@dataclass
class Neg(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return self._compile("-" + s.text)

@dataclass
class Desc(Sql):
    expr: Sql

    def compile(self):
        s = self.expr.compile()
        return self._compile(s.text + " DESC")


@dataclass
class Name(Sql):
    name: str

    def compile(self):
        return self._compile(self.name)

@dataclass
class ColumnAlias(Sql):
    value: Sql
    alias: Sql

    @classmethod
    def make(cls, value, alias):
        return cls(value.type, value, alias)

    def compile(self):
        alias = self.alias.compile().text
        value = self.value.compile().text
        if value == alias:
            s = alias  # This is just for beauty, it's not necessary for function
        else:
            s = '%s AS %s' % (value, alias)
        return self._compile(s)

@dataclass
class Insert(Sql):
    table: Table
    cols: List[str]
    values: List[Sql]

    def compile(self):

        q = ['INSERT INTO', self.table.name,
             "(", ', '.join(self.cols), ")",
             "VALUES",
             "(", ', '.join(v.compile().text for v in self.values), ")",
        ]
        insert = ' '.join(q) + ';'
        return self._compile(insert)

    # def import_result(self, value):
    #     assert not value
    #     return None

@dataclass
class LastRowId(Atom):
    type: types.PqlType = types.Int

    def compile(self):
        return self._compile('last_insert_rowid()')

@dataclass
class SelectValue(Atom):
    value: Sql

    def compile(self):
        value = self.value.compile()
        return self._compile(f'SELECT {value.text} as value')


class AllFields(Sql):
    def compile(self):
        return self._compile('*')

@dataclass
class Select(Table):
    table: Table
    fields: List[Sql]
    conds: List[Sql] = ()
    group_by: List[Sql] = ()
    order: List[Sql] = ()
    offset: Optional[Sql] = None
    limit: Optional[Sql] = None

    def __created__(self):
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

        return self._compile(sql)

    # def import_result(self, value):
    #     return self.type.from_sql_tuples(value)


@dataclass
class Join(Table):
    type: object
    tables: List[Table]
    conds: [Sql]
    join_op: str

    def compile(self):
        tables_sql = ['(%s)' % (t.compile().text) for t in self.tables]
        join_op = ' %s JOIN ' % self.join_op.upper()
        join_sql = join_op.join(e for e in tables_sql)

        join_sql += ' ON ' + ' AND '.join(c.compile().text for c in self.conds)

        return self._compile(join_sql)