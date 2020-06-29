from typing import List, Any, Optional, Dict

from .utils import dataclass, listgen, X
from . import exceptions as exc

from . import pql_types
from .pql_types import T, Type, Object


sqlite = 'sqlite'
postgres = 'postgres'

class QueryBuilder:
    def __init__(self, target, is_root=True, start_count=0, parameters=None):
        self.target = target
        self.is_root = is_root

        self.counter = start_count
        self.parameters = parameters or []

        self.table_name = []

    def unique_name(self):
        self.counter += 1
        return 't%d' % self.counter

    def replace(self, is_root):
        if is_root == self.is_root:
            return self # Optimize
        return QueryBuilder(self.target, is_root, self.counter, self.parameters)

    def push_table(self, t):
        self.table_name.append(t)
    def pop_table(self, t):
        t2 = self.table_name.pop()
        assert t2 == t

    def safe_name(self, base):
        "Return a name that is safe for use as variable. Must be consistent (pure func)"
        # if base.lower() in _reserved:
        if self.target == sqlite:
            return '[%s]' % base
        else:
            return '"%s"' % base



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
            if qb.is_root and self.type <= T.primitive:
                sql_code = f'SELECT {sql_code}'

        return CompiledSQL(sql_code, self)


@dataclass
class CompiledSQL:
    text: str
    sql: Sql

@dataclass
class RawSql(Sql):
    type: Type
    text: str

    def _compile(self, qb):
        return self.text

    @property
    def _is_select(self):
        return self.text.lower().startswith('select')   # XXX Hacky! Is there a cleaner solution?

@dataclass
class Null(Sql):
    type = T.null

    def _compile(self, qb):
        return 'null'

@dataclass
class Unknown(Sql):
    def _compile(self, qb):
        raise NotImplementedError("Unknown")

null = Null()
unknown = Unknown()

@dataclass
class Parameter(Sql):
    type: Type
    name: str

    def _compile(self, qb):
        # TODO messy
        from .evaluate import evaluate
        state, ns = qb.parameters[-1]
        obj = ns.get_var(state, self.name)
        obj = evaluate(state, obj)
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
    pass

@dataclass
class Atom(Scalar):
    pass

@dataclass
class Primitive(Atom):
    type: Type
    text: str

    def _compile(self, qb):
        return self.text


@dataclass
class Table(Sql):
    pass

@dataclass
class EmptyList(Table):
    type: Type

    _is_select = True

    def _compile(self, qb):
        return 'SELECT NULL AS VALUE LIMIT 0'


@dataclass
class TableName(Table):
    type: Type
    name: str

    def compile(self, qb):
        if qb.is_root:
            sql_code = f'SELECT * FROM {qb.safe_name(self.name)}'
        else:
            sql_code = qb.safe_name(self.name)

        return CompiledSQL(sql_code, self)

class TableOperation(Table):
    _is_select = True



@dataclass
class FieldFunc(Sql):
    name: str
    field: Sql
    type = T.int

    def _compile(self, qb):
        return f'{self.name}({self.field.compile(qb).text})'


@dataclass
class CountTable(Scalar):
    table: Sql
    type = T.int

    def _compile(self, qb):
        return f'(SELECT COUNT(*) FROM {self.table.compile(qb).text})'

@dataclass
class FuncCall(Sql):
    type: Type
    name: str
    fields: List[Sql]

    def _compile(self, qb):
        s = ', '.join(f.compile(qb).text for f in self.fields)
        return f'{self.name}({s})'

@dataclass
class Cast(Sql):
    type: Type
    as_type: str
    value: Sql

    def _compile(self, qb):
        return f'CAST({self.value.compile(qb).text} AS {self.as_type})'


@dataclass
class MakeArray(Sql):
    type: Type
    field: Sql

    _sp = "|"

    def _compile(self, qb):
        field = self.field.compile(qb).text
        if qb.target == sqlite:
            return f'group_concat({field}, "{self._sp}")'
        elif qb.target == postgres:
            return f'array_agg({field})'

        assert False, qb.target


@dataclass
class Contains(Scalar):
    op: str
    exprs: List[Sql]
    type = T.bool

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
    type = T.bool

    def __post_init__(self):
        assert self.op in ('=', '<=', '>=', '<', '>', '!=', 'IN'), self.op

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
        return '(%s)' % (f' {op} ').join(elems)

@dataclass
class Like(Scalar):
    string: Sql
    pattern: Sql
    type = T.bool

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
    type: Type
    name: str

    def __post_init__(self):
        assert self.name, self.type

    def _compile(self, qb):
        name = qb.safe_name(self.name)
        if qb.table_name:
            name = qb.table_name[-1] + '.' + name
        return name

    # return base

@dataclass
class ColumnAlias(Sql):
    value: Sql
    alias: str

    @classmethod
    def make(cls, value, alias):
        return cls(value, alias)

    def _compile(self, qb):
        alias = qb.safe_name(self.alias)
        value = self.value.compile(qb).text
        assert alias and value, (alias, value)
        if value == alias:  # TODO disable when unoptimized?
            return alias  # This is just for beauty, it's not necessary for function

        return f'{value} AS {alias}'

    type = property(X.value.type)


@dataclass
class Insert(Sql):
    table_name: str
    columns: List[str]
    query: Sql
    type = T.null

    def _compile(self, qb):
        return f'INSERT INTO "{self.table_name}"({", ".join(self.columns)}) SELECT * FROM ' + self.query.compile(qb).text

@dataclass
class InsertConsts(Sql):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.null

    def _compile(self, qb):
        assert self.tuples, self

        values = ', '.join(
            '(%s)' % ', '.join([e.compile(qb).text for e in tpl])
            for tpl in self.tuples
        )

        q = ['INSERT INTO', qb.safe_name(self.table),
             "(", ', '.join(self.cols), ")",
             "VALUES",
             values,
        ]
        return ' '.join(q) + ';'

@dataclass
class InsertConsts2(Sql):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.null

    def _compile(self, qb):
        assert self.tuples, self

        values = ', '.join(
            '(%s)' % ', '.join(tpl)
            for tpl in self.tuples
        )

        q = ['INSERT INTO', qb.safe_name(self.table),
             "(", ', '.join(self.cols), ")",
             "VALUES",
             values,
        ]
        return ' '.join(q) + ';'


@dataclass
class LastRowId(Atom):
    type = T.int

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
    type: Type
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile(qb) for v in self.values]
        if not values:  # SQL doesn't support empty values
            return 'SELECT NULL LIMIT 0'
        return 'VALUES' + ','.join(f'({v.text})' for v in values)


@dataclass
class AllFields(Sql):
    type: Type

    def _compile(self, qb):
        return '*'

@dataclass
class Update(Sql):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]
    type = T.null

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
    type = T.null

    def _compile(self, qb):
        conds = ' AND '.join(c.compile(qb).text for c in self.conds)
        return f'DELETE FROM {self.table.compile(qb).text} WHERE {conds}'

@dataclass
class Select(TableOperation):
    type: Type
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
    type: Type
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
        compare = Compare('=', [Name(T.int, 'id'), Primitive(T.int, str(id_))])
        yield Delete(TableName(table.type, table.type.options['name']), [compare])

def updates_by_ids(table, proj, ids):
    sql_proj = {Name(value.type, name): value.code for name, value in proj.items()}
    for id_ in ids:
        compare = Compare('=', [Name(T.int, 'id'), Primitive(T.int, str(id_))])
        yield Update(TableName(table.type, table.type.options['name']), sql_proj, [compare])

def create_list(list_type, name, elems):
    fields = [Name(list_type.elem, 'value')]
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

def arith(res_type, op, args):
    arg_codes = list(args)
    if res_type == T.string:
        assert op == '+'
        op = '||'
    elif op == '/':
        arg_codes[0] = Cast(T.float, 'float', arg_codes[0])
    elif op == '/~':
        op = '/'

    return Arith(op, arg_codes)


@dataclass
class StringSlice(Sql):
    string: Sql
    start: Sql
    stop: Optional[Sql]

    type = T.string

    def _compile(self, qb):
        string = self.string.compile(qb).text
        start = self.start.compile(qb).text
        if self.stop:
            stop = self.stop.compile(qb).text
            length = f'({stop}-{start})'
        else:
            length = None

        if qb.target == sqlite:
            f = 'substr'
            params = [string, ',', start]
            if length:
                params += [',', length]
        elif qb.target == postgres:
            f = 'substring'
            params = [string, 'from', start]
            if length:
                params += ['for', length]
        else:
            assert False

        return f'{f}({" ".join(params)})'


def value(x):
    if x is None:
        return null

    t = pql_types.from_python(type(x))

    if t <= T.datetime:
        # TODO Better to pass the object instead of a string?
        r = repr(str(x))

    elif t <= T.union[T.string, T.text]:
        r = "'%s'" % str(x).replace("'", "''")

    elif t <= T.decimal:
        r = repr(float(x))  # TODO SQL decimal?

    else:
        assert t <= T.union[T.number, T.bool], t
        r = repr(x)

    return Primitive(t, r)



from .pql_types import T, join_names, pql_dp, flatten_type, Type, Object, combined_dp, table_to_struct
def compile_type_def(state, table_name, table) -> Sql:
    assert table <= T.table

    posts = []
    pks = []
    columns = []

    pks = {join_names(pk) for pk in table.options['pk']}
    # autocount = types.join_names(table.autocount)
    for name, c in flatten_type(table):
        if name in pks:
            assert c <= T.t_id
            if state.db.target == postgres:
                type_ = "SERIAL" # Postgres
            else:
                type_ = "INTEGER"   # TODO non-int idtypes
        else:
            type_ = compile_type(c)

        columns.append( f'"{name}" {type_}' )
        if (c <= T.t_relation):
            # TODO any column, using projection / get_attr
            if not table.options.get('temporary', False):
                # In postgres, constraints on temporary tables may reference only temporary tables
                s = f"FOREIGN KEY({name}) REFERENCES \"{c.options['name']}\"(id)"
                posts.append(s)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    command = "CREATE TEMPORARY TABLE" if table.options.get('temporary', False) else "CREATE TABLE IF NOT EXISTS"
    return RawSql(T.null, f'{command} "{table_name}" (' + ', '.join(columns + posts) + ')')

@combined_dp
def compile_type(type_: T.t_relation):
    # TODO might have a different type
    return 'INTEGER'    # Foreign-key is integer

@combined_dp
def compile_type(type: T.primitive):
    assert type <= T.primitive
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'text': "TEXT",
        't_relation': "INTEGER",
        'datetime': "TIMESTAMP",
    }[type.typename]
    if not type.nullable:
        s += " NOT NULL"
    return s

@combined_dp
def compile_type(type: T.null):
    return 'INTEGER'    # TODO is there a better value here? Maybe make it read-only somehow

@combined_dp
def compile_type(idtype: T.t_id):
    s = "INTEGER"   # TODO non-int idtypes
    if not idtype.nullable:
        s += " NOT NULL"
    return s


