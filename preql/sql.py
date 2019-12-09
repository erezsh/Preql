from typing import List, Any, Optional, Dict

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
class Scalar(Sql):
    def import_result(self, res):
        row ,= res
        item ,= row
        return item

@dataclass
class Atom(Scalar):
    pass

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
            assert len(row) == expected_length, (expected_length, row)
            i = iter(row)
            s = ({str(name): col.type.restructure_result(i) for name, col in self.type.columns.items()})
            yield s

@dataclass
class TableName(Table):
    name: str

    def compile(self):
        return self._compile(self.name)

@dataclass
class FieldFunc(Sql):
    name: str
    field: Sql

    def compile(self):
        assert self.type is types.Int
        return self._compile(f'{self.name}({self.field.compile().text})')


@dataclass
class CountTable(Scalar):
    table: Table

    def compile(self):
        return self._compile(f'select count(*) from ({self.table.compile().text})')


@dataclass
class FuncCall(Sql):
    name: str
    fields: List[Sql]

    def compile(self):
        s = ', '.join(f.compile().text for f in self.fields)
        return self._compile(f'{self.name}({s})')

@dataclass
class Round(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self):
        return self._compile(f'round({self.field.compile().text})')

@dataclass
class Cast(Sql):
    as_type: str
    value: Sql

    def compile(self):
        return self._compile(f'CAST({self.value.compile().text} AS {self.as_type})')


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
class Arith(Scalar):
    op: str
    exprs: List[Sql]

    def compile(self):
        x = (' %s ' % self.op).join(e.compile().text for e in self.exprs)
        return self._compile('(%s)'%x)


@dataclass
class TableArith(Table):
    op: str
    exprs: List[Table]

    def compile(self):
        # XXX Limit -1 is due to a strange bug in SQLite (fixed in newer versions), where the limit is reset otherwise.
        tables = [t.compile() for t in self.exprs]
        selects = [f"SELECT * FROM ({t.text})" for t in tables]
        code = f" {self.op} ".join(selects) + " LIMIT -1"
        return self._compile(code)

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
    table_name: str
    query: Sql

    def compile(self):
        q = f'INSERT INTO {self.table_name} ' + self.query.compile().text
        return self._compile(q)


@dataclass
class InsertConsts(Sql):
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
class SelectValue(Atom, Table):
    value: Sql

    def compile(self):
        value = self.value.compile()
        return self._compile(f'SELECT {value.text} as value')

@dataclass
class Values(Table):
    values: List[Sql]

    def compile(self):
        values = [v.compile() for v in self.values]
        code = 'VALUES' + ','.join(f'({v.text})' for v in values)
        return self._compile(code)

class AllFields(Sql):
    def compile(self):
        return self._compile('*')

@dataclass
class Update(Sql):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]

    def compile(self):
        fields_sql = ['%s = %s' % (k.compile().text, v.compile().text) for k, v in self.fields.items()]
        fields_sql = ', '.join(fields_sql)

        sql = f'UPDATE {self.table.compile().text} SET {fields_sql}'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile().text for c in self.conds)

        return self._compile(sql)

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
    join_op: str
    tables: List[Table]
    conds: List[Sql]

    def compile(self):
        tables_sql = ['(%s)' % (t.compile().text) for t in self.tables]
        join_op = ' %s ' % self.join_op.upper()
        join_sql = join_op.join(e for e in tables_sql)

        if self.conds:
            join_sql += ' ON ' + ' AND '.join(c.compile().text for c in self.conds)

        join_sql = 'SELECT * FROM ' + join_sql
        return self._compile(join_sql)