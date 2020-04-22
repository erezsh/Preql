from typing import List, Any, Optional, Dict

from .utils import dataclass, listgen, X
from . import pql_types as types
from . import exceptions as exc

PqlType = types.PqlType


sqlite = 'sqlite'
postgres = 'postgres'

class QueryBuilder:
    def __init__(self, target, is_root=True, start_count=0, parameters=None):
        self.target = target
        self.is_root = is_root

        self.counter = start_count
        self.parameters = parameters or []

    def unique_name(self):
        self.counter += 1
        return 't%d' % self.counter

    def replace(self, is_root):
        if is_root == self.is_root:
            return self # Optimize
        return QueryBuilder(self.target, is_root, self.counter, self.parameters)


@dataclass
class Sql:
    _is_select = False

    def compile(self, qb):  # Move to Expr? Doesn't apply to statements
        sql_code = self._compile(qb.replace(is_root=False))
        assert isinstance(sql_code, str), self

        if self._is_select:
            if not qb.is_root:
                if qb.target == 'postgres':
                    sql_code = f'({sql_code}) {qb.unique_name()}' # postgres requires an alias
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
    type: PqlType
    text: str

    def _compile(self, qb):
        return self.text

    @property
    def _is_select(self):
        return self.text.lower().startswith('select')   # XXX Hacky! Is there a cleaner solution?

@dataclass
class Null(Sql):
    type = types.null

    def _compile(self, qb):
        return 'null'

null = Null()

@dataclass
class Parameter(Sql):
    type: PqlType
    name: str

    def _compile(self, qb):
        obj = qb.parameters[-1].get_var(self.name)
        assert obj.type == self.type
        return obj.code.compile(qb).text


@dataclass
class ResolveParameters(Sql):
    obj: Sql
    # values: Dict[str, Sql]
    ns: object

    def compile(self, qb):
        qb.parameters.append(self.ns)
        obj = self.obj.compile(qb)
        qb.parameters.pop()
        return obj

    @property
    def type(self):
        return self.obj.type


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
    type: PqlType
    text: str

    def _compile(self, qb):
        return self.text


@dataclass
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
    type: PqlType

    _is_select = True

    def _compile(self, qb):
        return 'SELECT NULL LIMIT 0'


@dataclass
class TableName(Table):
    type: PqlType
    name: str

    # def _compile(self, qb):
    #     return self.name

    def compile(self, qb):
        if qb.is_root:
            sql_code = f'SELECT * FROM {_safe_name(self.name)}'
        else:
            sql_code = _safe_name(self.name)

        return CompiledSQL(sql_code, self)

class TableOperation(Table):
    _is_select = True



@dataclass
class FieldFunc(Sql):
    name: str
    field: Sql
    type = types.Int

    def _compile(self, qb):
        return f'{self.name}({self.field.compile(qb).text})'


@dataclass
class CountTable(Scalar):
    table: Sql
    type = types.Int

    def _compile(self, qb):
        return f'(SELECT COUNT(*) FROM {self.table.compile(qb).text})'


@dataclass
class FuncCall(Sql):
    type: PqlType
    name: str
    fields: List[Sql]

    def _compile(self, qb):
        s = ', '.join(f.compile(qb).text for f in self.fields)
        return f'{self.name}({s})'

@dataclass
class Cast(Sql):
    type: PqlType
    as_type: str
    value: Sql

    def _compile(self, qb):
        return f'CAST({self.value.compile(qb).text} AS {self.as_type})'


@dataclass
class MakeArray(Sql):
    type: PqlType
    field: Sql

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
    type = types.Bool

    def _compile(self, qb):
        assert self.op
        item, container = self.exprs
        c_item = item.compile(qb).text
        c_cont = container.compile(qb.replace(is_root=True)).text
        return f'{c_item} {self.op} ({c_cont})'


@dataclass
class Compare(Scalar):
    op: str
    exprs: List[Sql]
    type = types.Bool

    def __post_init__(self):
        assert self.op in ('=', '<=', '>=', '<', '>', '!='), self.op

    def _compile(self, qb):
        op = self.op
        if qb.target == sqlite:
            op = {
                '=': 'is',
                '!=': 'is not'
            }.get(op, op)
        else:
            op = {
                '=': 'is not distinct from',
                '!=': 'is distinct from'
            }.get(op, op)

        elems = [e.compile(qb).text for e in self.exprs]
        return (f' {op} ').join(elems)

@dataclass
class Like(Scalar):
    string: Scalar
    pattern: Scalar
    type = types.Bool

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

    type = property(X.exprs[0].type)     # TODO ensure type correctness


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

    type = property(X.exprs[0].type)   # TODO ensure type correctness


@dataclass
class Neg(Sql):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return "-" + s.text

    type = property(X.expr.type)

@dataclass
class Desc(Sql):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return s.text + " DESC"

    type = property(X.expr.type)

_reserved = {'index', 'create', 'unique', 'table', 'select', 'where', 'group', 'by', 'over', 'user'}

@dataclass
class Name(Sql):
    type: PqlType
    name: str

    def __post_init__(self):
        assert self.name, self.type

    def _compile(self, qb):
        return _safe_name(self.name)

def _safe_name(base):
    "Return a name that is safe for use as variable. Must be consistent (pure func)"
    # if base.lower() in _reserved:
    return '"%s"' % base
    # return base

@dataclass
class ColumnAlias(Sql):
    value: Sql
    alias: str

    @classmethod
    def make(cls, value, alias):
        return cls(value, alias)

    def _compile(self, qb):
        alias = _safe_name(self.alias)
        value = self.value.compile(qb).text
        assert alias and value, (alias, value)
        if value == alias:  # TODO disable when unoptimized?
            return alias  # This is just for beauty, it's not necessary for function

        return f'{value} AS {alias}'

    type = property(X.value.type)


@dataclass
class Insert(Sql):
    table_type: types.TableType
    columns: List[str]
    query: Sql
    type = types.null

    def _compile(self, qb):
        return f'INSERT INTO "{self.table_type.name}"({", ".join(self.columns)}) SELECT * FROM ' + self.query.compile(qb).text

@dataclass
class InsertConsts(Sql):
    table: Sql
    cols: List[str]
    values: List[Sql]
    type = types.null

    def _compile(self, qb):
        assert self.values

        q = ['INSERT INTO', _safe_name(self.table.name),
             "(", ', '.join(self.cols), ")",
             "VALUES",
             "(", ', '.join(v.compile(qb).text for v in self.values), ")",
        ]
        return ' '.join(q) + ';'


@dataclass
class LastRowId(Atom):
    type = types.Int

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

    type = property(X.value.type)

@dataclass
class RowDict(Sql):
    values: Dict[str, Sql]

    def _compile(self, qb):
        return {f'{v.compile(qb).text} as {name}' for name, v in self.values.items()}


@dataclass
class Values(Table):
    type: PqlType
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile(qb) for v in self.values]
        if not values:  # SQL doesn't support empty values
            return 'SELECT NULL LIMIT 0'
        return 'VALUES' + ','.join(f'({v.text})' for v in values)


@dataclass
class AllFields(Sql):
    type: PqlType

    def _compile(self, qb):
        return '*'

@dataclass
class Update(Sql):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]
    type = types.null

    def _compile(self, qb):
        fields_sql = ['%s = %s' % (k.compile(qb).text, v.compile(qb).text) for k, v in self.fields.items()]
        fields_sql = ', '.join(fields_sql)

        sql = f'UPDATE {self.table.compile(qb).text} SET {fields_sql}'

        if self.conds:
            sql += ' WHERE ' + ' AND '.join(c.compile(qb).text for c in self.conds)

        return sql

@dataclass
class Delete(Sql):
    table: TableName
    conds: List[Sql]
    type = types.null

    def _compile(self, qb):
        conds = ' AND '.join(c.compile(qb).text for c in self.conds)
        return f'DELETE FROM {self.table.compile(qb).text} WHERE {conds}'

@dataclass
class Select(TableOperation):
    type: PqlType
    table: Sql # XXX Table won't work with RawSQL
    fields: List[Sql]
    conds: List[Sql] = ()
    group_by: List[Sql] = ()
    order: List[Sql] = ()
    offset: Optional[Sql] = None
    limit: Optional[Sql] = None

    def __post_init__(self):
        assert self.fields, self

    def _is_conds_only(self):
        if self.group_by or self.order or self.offset or self.limit:
            return False

        if len(self.fields) == 1 and isinstance(self.fields[0], AllFields):
            return True

        return False

    def _compile(self, qb):
        # XXX very primitive optimization. Be smarter.
        #
        # Simplify
        #
        if isinstance(self.table, Select):
            s1 = self
            s2 = self.table
            if s2._is_conds_only():
                s = s1.replace(conds=list(s1.conds) + list(s2.conds), table=s2.table)
                return s._compile(qb)

        #
        # Compile
        #

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
                sql += ' LIMIT -1'  # Sqlite only (and only old versions of it)

        if self.offset:
            sql += ' OFFSET ' + self.offset.compile(qb).text

        if self.order:
            sql += ' ORDER BY ' + ', '.join(o.compile(qb).text for o in self.order)

        return sql


@dataclass
class Subquery(Sql):
    # type: PqlType
    table_name: str
    fields: List[Name]
    query: Sql
    type = property(X.query.type)

    def _compile(self, qb):
        query = self.query.compile(qb).text
        fields = [f.compile(qb.replace(is_root=False)).text for f in self.fields]
        return f"{self.table_name}({', '.join(fields)}) AS ({query})"

    def compile(self, qb):
        sql_code = self._compile(qb)
        return CompiledSQL(sql_code, self)



@dataclass
class Join(TableOperation):
    type: PqlType
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





def deletes_by_ids(table, ids):
    for id_ in ids:
        compare = Compare('=', [Name(types.Int, 'id'), Primitive(types.Int, str(id_))])
        yield Delete(TableName(table.type, table.type.name), [compare])

def updates_by_ids(table, proj, ids):
    sql_proj = {Name(value.type, name): value.code for name, value in proj.items()}
    for id_ in ids:
        compare = Compare('=', [Name(types.Int, 'id'), Primitive(types.Int, str(id_))])
        yield Update(TableName(table.type, table.type.name), sql_proj, [compare])

def create_list(list_type, name, elems):
    fields = [Name(list_type.elemtype, 'value')]
    subq = Subquery(name, fields, Values(list_type, elems))
    table = TableName(list_type, name)
    return table, subq

def table_slice(table, start, stop):
    limit = Arith('-', [stop, start]) if stop else None
    return Select(table.type, table.code, [AllFields(table.type)], offset=start, limit=limit)

def table_selection(table, conds):
    return Select(table.type, table.code, [AllFields(table.type)], conds=conds)

def table_order(table, fields):
    return Select(table.type, table.code, [AllFields(table.type)], order=fields)

def arith(res_type, op, args, meta):
    arg_codes = list(args)
    if res_type == types.String:
        if op != '+':
            meta = op.meta.replace(parent=meta)
            raise exc.pql_TypeError(meta, f"Operator '{op}' not supported for strings.")
        op = '||'
    elif op == '/':
        arg_codes[0] = Cast(types.Float, 'float', arg_codes[0])
    elif op == '/~':
        op = '/'

    return Arith(op, arg_codes)




def value(x):
    if x is None:
        return null

    t = types.Primitive.by_pytype[type(x)]

    if t is types.DateTime:
        # TODO Better to pass the object instead of a string?
        return Primitive(t, repr(str(x)))

    if t is types.String or t is types.Text:
        return Primitive(t, "'%s'" % str(x).replace("'", "''"))

    return Primitive(t, repr(x))
