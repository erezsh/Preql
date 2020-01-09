from typing import List, Any, Optional, Dict

from .utils import dataclass, SafeDict, listgen
from . import pql_types as types


sqlite = 'sqlite'
postgres = 'postgres'

class QueryBuilder:
    def __init__(self, target, is_root=True, start_count=0):
        self.target = target
        self.is_root = is_root

        self.counter = start_count

    def get_alias(self):
        self.counter += 1
        return 't%d' % self.counter

    def remake(self, is_root):
        if is_root == self.is_root:
            return self # Optimize
        return QueryBuilder(self.target, is_root, self.counter)


@dataclass
class Sql:
    type: types.PqlType

    _is_select = False

    def compile(self, qb):  # Move to Expr? Doesn't apply to statements
        sql_code = self._compile(qb.remake(is_root=False))
        assert isinstance(sql_code, str)

        if self._is_select:
            if not qb.is_root:
                if qb.target == 'postgres':
                    sql_code = f'({sql_code}) {qb.get_alias()}' # postgres requires an alias
                else:
                    sql_code = f'({sql_code})'
        else:
            if qb.is_root and isinstance(self.type, types.Primitive):
                sql_code = f'SELECT {sql_code}'

        return CompiledSQL(sql_code, self)


@dataclass
class CompiledSQL:
    text: str
    sql: Sql

@dataclass
class RawSql(Sql):
    text: str

    def _compile(self, qb):
        return self.text

    @property
    def _is_select(self):
        return self.text.lower().startswith('select')   # XXX Hacky! Is there a cleaner solution?

@dataclass
class Null(Sql):
    def _compile(self, qb):
        return 'null'

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

    def _compile(self, qb):
        return self.text


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
class EmptyList(Table):
    _is_select = True

    def _compile(self, qb):
        return 'SELECT NULL LIMIT 0'


@dataclass
class TableName(Table):
    name: str

    # def _compile(self, qb):
    #     return self.name

    def compile(self, qb):
        if qb.is_root:
            sql_code = f'SELECT * FROM {self.name}'
        else:
            sql_code = self.name

        return CompiledSQL(sql_code, self)

class TableOperation(Table):
    _is_select = True



@dataclass
class FieldFunc(Sql):
    name: str
    field: Sql

    def _compile(self, qb):
        assert self.type is types.Int
        return f'{self.name}({self.field.compile(qb).text})'


@dataclass
class CountTable(Scalar):
    table: Table

    def _compile(self, qb):
        return f'(SELECT COUNT(*) FROM {self.table.compile(qb).text})'


@dataclass
class FuncCall(Sql):
    name: str
    fields: List[Sql]

    def _compile(self, qb):
        s = ', '.join(f.compile(qb).text for f in self.fields)
        return f'{self.name}({s})'

@dataclass
class Cast(Sql):
    as_type: str
    value: Sql

    def _compile(self, qb):
        return f'CAST({self.value.compile(qb).text} AS {self.as_type})'


@dataclass
class MakeArray(Sql):
    field: Sql
    type = list  # TODO correct object

    _sp = "|"

    def _compile(self, qb):
        field = self.field.compile(qb).text
        if qb.target == sqlite:
            return f'group_concat({field}, "{self._sp}")'
        elif qb.target == postgres:
            return f'array_agg({field})'

        assert False, qb.target

    def import_result(self, value):
        assert False
        if value is None:
            return []
        return value.split(self._sp)


@dataclass
class Contains(Scalar):
    op: str
    exprs: List[Sql]

    def _compile(self, qb):
        assert self.op
        item, container = self.exprs
        c_item = item.compile(qb).text
        c_cont = container.compile(qb.remake(is_root=True)).text
        return f'{c_item} {self.op} ({c_cont})'


@dataclass
class Compare(Scalar):
    op: str
    exprs: List[Sql]

    def __created__(self):
        assert self.op in ('=', '<=', '>=', '<', '>', '<>', '!='), self.op

    def _compile(self, qb):
        elems = [e.compile(qb).text for e in self.exprs]
        return (f' {self.op} ').join(elems)

@dataclass
class Like(Scalar):
    string: Scalar
    pattern: Scalar

    def _compile(self, qb):
        s = self.string.compile(qb)
        p = self.pattern.compile(qb)
        return f'{s.text} like {p.text}'

@dataclass
class Arith(Scalar):
    op: str
    exprs: List[Sql]

    def _compile(self, qb):
        x = (f' {self.op} ').join(e.compile(qb).text for e in self.exprs)
        return f'({x})'


@dataclass
class TableArith(TableOperation):
    op: str
    exprs: List[Table]

    def _compile(self, qb):
        tables = [t.compile(qb) for t in self.exprs]
        selects = [f"SELECT * FROM {t.text}" for t in tables]

        code = f" {self.op} ".join(selects)

        if qb.target == sqlite:
            # Limit -1 is due to a strange bug in SQLite (fixed in newer versions), where the limit is reset otherwise.
            code += " LIMIT -1"

        return code


@dataclass
class Neg(Sql):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return "-" + s.text

@dataclass
class Desc(Sql):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return s.text + " DESC"

_reserved = {'index', 'create', 'unique', 'table', 'select', 'where', 'group', 'by', 'over'}

@dataclass
class Name(Sql):
    name: str

    def _compile(self, qb):
        if self.name.lower() in _reserved:
            return self.name + "_"
        return self.name

@dataclass
class ColumnAlias(Sql):
    value: Sql
    alias: Sql

    @classmethod
    def make(cls, value, alias):
        return cls(value.type, value, alias)

    def _compile(self, qb):
        alias = self.alias.compile(qb).text
        value = self.value.compile(qb).text
        assert alias and value
        if value == alias:  # TODO disable when unoptimized?
            return alias  # This is just for beauty, it's not necessary for function

        return f'{value} AS {alias}'


@dataclass
class Insert(Sql):
    table_name: str
    query: Sql

    def _compile(self, qb):
        return f'INSERT INTO {self.table_name} SELECT * FROM ' + self.query.compile(qb).text


@dataclass
class InsertConsts(Sql):
    table: Table
    cols: List[str]
    values: List[Sql]

    def _compile(self, qb):

        q = ['INSERT INTO', self.table.name,
             "(", ', '.join(self.cols), ")",
             "VALUES",
             "(", ', '.join(v.compile(qb).text for v in self.values), ")",
        ]
        return ' '.join(q) + ';'


@dataclass
class LastRowId(Atom):
    type: types.PqlType = types.Int

    def _compile(self, qb):
        if qb.target == sqlite:
            return 'last_insert_rowid()'   # Sqlite
        else:
            return 'lastval()'   # Postgres

@dataclass
class SelectValue(Atom, TableOperation):
    # XXX Just use a regular select?
    value: Sql

    def _compile(self, qb):
        value = self.value.compile(qb)
        return f'SELECT {value.text} AS value'

@dataclass
class Values(Table):
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile(qb) for v in self.values]
        return 'VALUES' + ','.join(f'({v.text})' for v in values)

class AllFields(Sql):
    def _compile(self, qb):
        return '*'

@dataclass
class Update(Sql):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]

    def _compile(self, qb):
        fields_sql = ['%s = %s' % (k.compile(qb).text, v.compile(qb).text) for k, v in self.fields.items()]
        fields_sql = ', '.join(fields_sql)

        sql = f'UPDATE {self.table.compile(qb).text} SET {fields_sql}'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile(qb).text for c in self.conds)

        return sql

@dataclass
class Select(TableOperation):
    table: Sql # XXX Table won't work with RawSQL
    fields: List[Sql]
    conds: List[Sql] = ()
    group_by: List[Sql] = ()
    order: List[Sql] = ()
    offset: Optional[Sql] = None
    limit: Optional[Sql] = None

    def __created__(self):
        assert self.fields, self

    def _compile(self, qb):
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
            if qb.target == sqlite:
                sql += ' LIMIT -1'  # Sqlite only

        if self.offset:
            sql += ' OFFSET ' + self.offset.compile(qb).text

        if self.order:
            sql += ' ORDER BY ' + ', '.join(o.compile(qb).text for o in self.order)

        return sql

@dataclass
class Subquery(Sql):
    table_name: str
    fields: List[Name]
    query: Sql

    def _compile(self, qb):
        query = self.query.compile(qb).text
        fields = [f.compile(qb.remake(is_root=False)).text for f in self.fields]
        return f"{self.table_name}({', '.join(fields)}) AS ({query})"

    def compile(self, qb):
        sql_code = self._compile(qb)
        return CompiledSQL(sql_code, self)


@dataclass
class Join(TableOperation):
    join_op: str
    tables: List[Table]
    conds: List[Sql]

    def _compile(self, qb):
        tables_sql = [t.compile(qb).text for t in self.tables]
        join_op = ' %s ' % self.join_op.upper()
        join_sql = join_op.join(e for e in tables_sql)

        if self.conds:
            conds = ' AND '.join(c.compile(qb).text for c in self.conds)
        else:
            conds = '1=1'   # Postgres requires ON clause

        return f'SELECT * FROM {join_sql} ON {conds}'