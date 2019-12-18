from typing import List, Any, Optional, Dict

from .utils import dataclass, SafeDict, listgen
from . import pql_types as types


sqlite = 'sqlite'
postgres = 'postgres'

class QueryBuilder:
    def __init__(self, target, is_root=True):
        self.target = target
        self.is_root = is_root

        self.counter = 0

    def get_alias(self):
        self.counter += 1
        return 't%d' % self.counter



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

    def compile(self, qb):
        return self._compile(self.text)

@dataclass
class Null(Sql):
    def compile(self, qb):
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

    def compile(self, qb):
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

    def compile(self, qb):
        return self._compile(self.name)

@dataclass
class FieldFunc(Sql):
    name: str
    field: Sql

    def compile(self, qb):
        assert self.type is types.Int
        return self._compile(f'{self.name}({self.field.compile(qb).text})')


@dataclass
class CountTable(Scalar):
    table: Table

    def compile(self, qb):
        return self._compile(f'select count(*) from ({self.table.compile(qb).text})')


@dataclass
class FuncCall(Sql):
    name: str
    fields: List[Sql]

    def compile(self, qb):
        s = ', '.join(f.compile(qb).text for f in self.fields)
        return self._compile(f'{self.name}({s})')

@dataclass
class Round(Sql):
    field: Sql
    type = float  # TODO correct object

    def compile(self, qb):
        return self._compile(f'round({self.field.compile(qb).text})')

@dataclass
class Cast(Sql):
    as_type: str
    value: Sql

    def compile(self, qb):
        return self._compile(f'CAST({self.value.compile(qb).text} AS {self.as_type})')


@dataclass
class MakeArray(Sql):
    field: Sql
    type = list  # TODO correct object

    _sp = "|"

    def compile(self, qb):
        if qb.target == sqlite:
            return self._compile(f'group_concat({self.field.compile(qb).text}, "{self._sp}")')
        elif qb.target == postgres:
            return self._compile(f'array_agg({self.field.compile(qb).text})')
        else:
            assert False

    def import_result(self, value):
        assert False
        if value is None:
            return []
        return value.split(self._sp)


@dataclass
class Contains(Sql):
    op: str
    exprs: List[Sql]

    def compile(self, qb):
        assert self.op
        qb.is_root = True
        a, b = ['%s' % e.compile(qb).text for e in self.exprs]
        contains = f'{a} {self.op} ({b})'
        return self._compile(contains)


@dataclass
class Compare(Sql):
    op: str
    exprs: List[Sql]

    def __created__(self):
        assert self.op in ('=', '<=', '>=', '<', '>', '<>', '!='), self.op

    def compile(self, qb):
        elems = [e.compile(qb).text for e in self.exprs]
        compare = (' %s ' % self.op).join(elems)
        return self._compile(compare)

@dataclass
class Arith(Scalar):
    op: str
    exprs: List[Sql]

    def compile(self, qb):
        x = (' %s ' % self.op).join(e.compile(qb).text for e in self.exprs)
        return self._compile('(%s)'%x)


@dataclass
class TableArith(Table):
    op: str
    exprs: List[Table]

    def compile(self, qb):
        is_root = qb.is_root
        qb.is_root = False

        # XXX Limit -1 is due to a strange bug in SQLite (fixed in newer versions), where the limit is reset otherwise.
        tables = [t.compile(qb) for t in self.exprs]
        selects = [f"SELECT * FROM {t.text}" for t in tables]

        if qb.target == sqlite:
            code = f" {self.op} ".join(selects) + " LIMIT -1"
        else:
            code = f" {self.op} ".join(selects)

        # return self._compile('(%s)' % code)
        if not is_root:
            code = '(%s) %s' % (code, qb.get_alias())
        return self._compile(code)


@dataclass
class Neg(Sql):
    expr: Sql

    def compile(self, qb):
        s = self.expr.compile(qb)
        return self._compile("-" + s.text)

@dataclass
class Desc(Sql):
    expr: Sql

    def compile(self, qb):
        s = self.expr.compile(qb)
        return self._compile(s.text + " DESC")


@dataclass
class Name(Sql):
    name: str

    def compile(self, qb):
        return self._compile(self.name)

@dataclass
class ColumnAlias(Sql):
    value: Sql
    alias: Sql

    @classmethod
    def make(cls, value, alias):
        return cls(value.type, value, alias)

    def compile(self, qb):
        alias = self.alias.compile(qb).text
        value = self.value.compile(qb).text
        if value == alias:
            s = alias  # This is just for beauty, it's not necessary for function
        else:
            s = '%s AS %s' % (value, alias)
        return self._compile(s)


@dataclass
class Insert(Sql):
    table_name: str
    query: Sql

    def compile(self, qb):
        q = f'INSERT INTO {self.table_name} ' + self.query.compile(qb).text
        return self._compile(q)


@dataclass
class InsertConsts(Sql):
    table: Table
    cols: List[str]
    values: List[Sql]

    def compile(self, qb):

        q = ['INSERT INTO', self.table.name,
             "(", ', '.join(self.cols), ")",
             "VALUES",
             "(", ', '.join(v.compile(qb).text for v in self.values), ")",
        ]
        insert = ' '.join(q) + ';'
        return self._compile(insert)


@dataclass
class LastRowId(Atom):
    type: types.PqlType = types.Int

    def compile(self, qb):
        if qb.target == sqlite:
            return self._compile('last_insert_rowid()')   # Sqlite
        else:
            return self._compile('lastval()')   # Postgres

@dataclass
class SelectValue(Atom, Table):
    value: Sql

    def compile(self, qb):
        is_root = qb.is_root
        qb.is_root = False

        value = self.value.compile(qb)
        code = f'SELECT {value.text} AS value'

        if not is_root:
            code = '(%s) %s' % (code, qb.get_alias())
        return self._compile(code)

@dataclass
class Values(Table):
    values: List[Sql]

    def compile(self, qb):
        values = [v.compile(qb) for v in self.values]
        code = 'VALUES' + ','.join(f'({v.text})' for v in values)
        return self._compile(code)

class AllFields(Sql):
    def compile(self, qb):
        return self._compile('*')

@dataclass
class Update(Sql):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]

    def compile(self, qb):
        fields_sql = ['%s = %s' % (k.compile(qb).text, v.compile(qb).text) for k, v in self.fields.items()]
        fields_sql = ', '.join(fields_sql)

        sql = f'UPDATE {self.table.compile(qb).text} SET {fields_sql}'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile(qb).text for c in self.conds)

        return self._compile(sql)

@dataclass
class Select(Table):
    table: Sql # XXX Table won't work with RawSQL
    fields: List[Sql]
    conds: List[Sql] = ()
    group_by: List[Sql] = ()
    order: List[Sql] = ()
    offset: Optional[Sql] = None
    limit: Optional[Sql] = None

    def __created__(self):
        assert self.fields, self

    def compile(self, qb):
        is_root = qb.is_root
        qb.is_root = False

        fields_sql = [f.compile(qb) for f in self.fields]
        select_sql = ', '.join(f.text for f in fields_sql)

        sql = f'SELECT {select_sql} FROM {self.table.compile(qb).text}'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile(qb).text for c in self.conds)

        if self.group_by:
            sql += ' GROUP BY ' + ', '.join(e.compile(qb).text for e in self.group_by)

        if self.limit:
            sql += ' LIMIT ' + self.limit.compile(qb).text
        elif self.offset:
            sql += ' LIMIT -1'  # XXX Sqlite only

        if self.offset:
            sql += ' OFFSET ' + self.offset.compile(qb).text

        if self.order:
            sql += ' ORDER BY ' + ', '.join(o.compile(qb).text for o in self.order)

        if not is_root:
            sql = '(%s) %s' % (sql, qb.get_alias())
        return self._compile(sql)

    # def import_result(self, value):
    #     return self.type.from_sql_tuples(value)


@dataclass
class Join(Table):
    join_op: str
    tables: List[Table]
    conds: List[Sql]

    def compile(self, qb):
        is_root = qb.is_root
        qb.is_root = False

        tables_sql = ['%s' % (t.compile(qb).text) for t in self.tables]
        join_op = ' %s ' % self.join_op.upper()
        join_sql = join_op.join(e for e in tables_sql)

        if self.conds:
            join_sql += ' ON ' + ' AND '.join(c.compile(qb).text for c in self.conds)
        else:
            join_sql += ' ON 1=1'   # Postgres requires ON clause

        join_sql = 'SELECT * FROM ' + join_sql
        # return self._compile('(%s) %s' % (join_sql, qb.get_alias() ))

        if not is_root:
            join_sql = '(%s) %s' % (join_sql, qb.get_alias())
        return self._compile(join_sql)