from typing import List, Optional, Dict

from .utils import dataclass, X, listgen
from . import pql_types
from .pql_types import T, join_names, flatten_type, Type, dp_type


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



class Sql:
    pass

@dataclass
class SqlTree(Sql):
    _is_select = False

    def compile(self, qb):  # Move to Expr? Doesn't apply to statements
        sql_code = self._compile(qb.replace(is_root=False))
        assert isinstance(sql_code, list), self
        assert all(isinstance(c, (str, Parameter)) for c in sql_code), self
        # assert sum(x.count('(') for x in sql_code) == sum(x.count(')') for x in sql_code)

        return CompiledSQL(self.type, sql_code, self, self._is_select, False).wrap(qb)  # XXX hack

    def compile_for_cache(self, qb):
        assert False
        sql_code = self._compile(qb.replace(is_root=False))
        assert isinstance(sql_code, list), self
        assert all(isinstance(c, (str, Parameter)) for c in sql_code), self
        # assert sum(x.count('(') for x in sql_code) == sum(x.count(')') for x in sql_code)

        return CompiledSQL(self.type, sql_code, self, self._is_select, False)


@dataclass
class CompiledSQL(Sql):
    type: Type
    code: list
    source_tree: Optional[Sql]
    _is_select: bool   # Needed for embedding in SqlTree
    _needs_select: bool

    def finalize(self, state, qb):
        assert qb.is_root
        if self.type <= T.primitive:
            code = ['SELECT '] + self.code
        else:
            code = self.code
        return ''.join(code)

    def wrap(self, qb):
        # XXX this shouldn't happen. Make it all compiled always!
        code = self.code

        if qb.is_root:
            if self._needs_select:
                code = ['SELECT * FROM'] + code
                return self.replace(code=code, _needs_select=False)
        else:
            if self._is_select and not self._needs_select:
                # Bad recursion
                if qb.target == 'postgres':
                    code = ['('] + code + [') ', qb.unique_name()]  # postgres requires an alias
                else:
                    code = ['('] + code + [')']

                return self.replace(code=code, _is_select=False)

        return self

    def compile(self, qb):
        return self
    def compile_for_cache(self, qb):
        return self

    def optimize(self):
        if not self.code:
            return self

        # unify strings for faster resolve_parameters and finalize
        new_code = []
        last = self.code[0]
        for c in self.code[1:]:
            if isinstance(c, str) and isinstance(last, str):
                last += c
            else:
                new_code.append(last)
                last = c
        new_code.append(last)
        return self.replace(code=new_code)


@dataclass
class RawSql(SqlTree):
    type: Type
    text: str

    def _compile(self, qb):
        return [self.text]

    @property
    def _is_select(self):
        return self.text.lstrip().lower().startswith('select')   # XXX Hacky! Is there a cleaner solution?

@dataclass
class Null(SqlTree):
    type = T.null

    def _compile(self, qb):
        return ['null']

@dataclass
class Unknown(SqlTree):
    def _compile(self, qb):
        raise NotImplementedError("Unknown")

null = Null()
unknown = Unknown()

@dataclass
class Parameter(SqlTree):
    type: Type
    name: str

    def _compile(self, qb):
        return [self]


@dataclass
class Scalar(SqlTree):
    pass

@dataclass
class Atom(Scalar):
    pass

@dataclass
class Primitive(Atom):
    type: Type
    text: str

    def _compile(self, qb):
        return [self.text]


@dataclass
class Table(SqlTree):
    pass

@dataclass
class EmptyList(Table):
    type: Type

    _is_select = True

    def _compile(self, qb):
        return ['SELECT NULL AS VALUE LIMIT 0']


@dataclass
class TableName(Table):
    type: Type
    name: str

    def compile(self, qb):
        sql_code = qb.safe_name(self.name)
        return CompiledSQL(self.type, [sql_code], self, True, True).wrap(qb)

    def compile_for_cache(self, qb):
        assert False
        sql_code = qb.safe_name(self.name)
        return CompiledSQL(self.type, [sql_code], self, True, True)


class TableOperation(Table):
    _is_select = True



@dataclass
class FieldFunc(SqlTree):
    name: str
    field: Sql
    type = T.int

    def _compile(self, qb):
        return [f'{self.name}('] + self.field.compile(qb).code + [')']


@dataclass
class CountTable(Scalar):
    table: Sql
    type = T.int

    def _compile(self, qb):
        return [f'(SELECT COUNT(*) FROM '] + self.table.compile(qb).code + [')']

@dataclass
class FuncCall(SqlTree):
    type: Type
    name: str
    fields: List[Sql]

    def _compile(self, qb):
        s = join_comma(f.compile(qb).code for f in self.fields)
        return [f'{self.name}('] + s + [')']

@dataclass
class Cast(SqlTree):
    type: Type
    as_type: str
    value: Sql

    def _compile(self, qb):
        return [f'CAST('] + self.value.compile(qb).code + [f' AS {self.as_type})']


@dataclass
class MakeArray(SqlTree):
    type: Type
    field: Sql

    _sp = "|"

    def _compile(self, qb):
        field = self.field.compile(qb).code
        if qb.target == sqlite:
            return ['group_concat('] + field + [f', "{self._sp}")']
        elif qb.target == postgres:
            return ['array_agg('] + field + [')']

        assert False, qb.target


@dataclass
class Contains(Scalar):
    op: str
    exprs: List[Sql]
    type = T.bool

    def _compile(self, qb):
        assert self.op
        item, container = self.exprs
        c_item = item.compile(qb).code
        c_cont = container.compile(qb.replace(is_root=True)).code
        return c_item + [' ', self.op, ' '] + parens(c_cont)


def parens(x):
    return ['('] + x + [')']

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

        elems = [e.compile(qb).code for e in self.exprs]
        return parens( join_sep(elems, f' {op} ') )

@dataclass
class Like(Scalar):
    string: Sql
    pattern: Sql
    type = T.bool

    def _compile(self, qb):
        s = self.string.compile(qb)
        p = self.pattern.compile(qb)
        return s.code + [' like '] + p.code

@dataclass
class Arith(Scalar):
    op: str
    exprs: List[Sql]

    def _compile(self, qb):
        x = join_sep([e.compile(qb).code for e in self.exprs], f' {self.op} ')
        return parens(x)

    type = property(X.exprs[0].type)     # TODO ensure type correctness


@dataclass
class TableArith(TableOperation):
    op: str
    exprs: List[Table]

    def _compile(self, qb):
        tables = [t.compile(qb) for t in self.exprs]
        selects = [[f"SELECT * FROM "] + t.code for t in tables]

        code = join_sep(selects, f" {self.op} ")

        if qb.target == sqlite:
            # Limit -1 is due to a strange bug in SQLite (fixed in newer versions), where the limit is reset otherwise.
            code += [" LIMIT -1"]

        return code

    type = property(X.exprs[0].type)   # TODO ensure type correctness


@dataclass
class Neg(SqlTree):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return ["-"] + s.code

    type = property(X.expr.type)

@dataclass
class Desc(SqlTree):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile(qb)
        return s.code + [" DESC"]

    type = property(X.expr.type)

_reserved = {'index', 'create', 'unique', 'table', 'select', 'where', 'group', 'by', 'over', 'user'}

@dataclass
class Name(SqlTree):
    type: Type
    name: str

    def __post_init__(self):
        assert self.name, self.type

    def _compile(self, qb):
        name = qb.safe_name(self.name)
        if qb.table_name:
            name = qb.table_name[-1] + '.' + name
        return [name]

@dataclass
class Attr(SqlTree):
    type: Type
    obj: Sql
    name: str



    # return base

@dataclass
class ColumnAlias(SqlTree):
    value: Sql
    alias: str

    @classmethod
    def make(cls, value, alias):
        return cls(value, alias)

    def _compile(self, qb):
        alias = qb.safe_name(self.alias)
        value = self.value.compile(qb).code
        assert alias and value, (alias, value)
        if value == alias:  # TODO disable when unoptimized?
            return alias  # This is just for beauty, it's not necessary for function

        return value + [f' AS {alias}']

    type = property(X.value.type)


@dataclass
class Insert(SqlTree):
    table_name: str
    columns: List[str]
    query: Sql
    type = T.null

    def _compile(self, qb):
        return [f'INSERT INTO "{self.table_name}"({", ".join(self.columns)}) SELECT * FROM '] + self.query.compile(qb).code

@dataclass
class InsertConsts(SqlTree):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.null

    def _compile(self, qb):
        assert self.tuples, self

        values = join_comma(
            parens(join_comma([e.compile(qb).code for e in tpl]))
            for tpl in self.tuples
        )

        q = ['INSERT INTO', qb.safe_name(self.table),
             "(", ', '.join(self.cols), ")",
             "VALUES ",
        ]
        return [' '.join(q)] + values #+ [';']

@dataclass
class InsertConsts2(SqlTree):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.null

    def _compile(self, qb):
        assert self.tuples, self

        values = join_comma(
            parens(join_comma(tpl))
            for tpl in self.tuples
        )

        q = ['INSERT INTO', qb.safe_name(self.table),
             "(", ', '.join(self.cols), ")",
             "VALUES ",
        ]
        return [' '.join(q)] + values #+ ';'


@dataclass
class LastRowId(Atom):
    type = T.int

    def _compile(self, qb):
        if qb.target == sqlite:
            return ['last_insert_rowid()']   # Sqlite
        else:
            return ['lastval()']   # Postgres

@dataclass
class SelectValue(Atom, TableOperation):
    # XXX Just use a regular select?
    value: Sql

    def _compile(self, qb):
        value = self.value.compile(qb)
        return [f'SELECT '] + value.code + [' AS '] + value

    type = property(X.value.type)

@dataclass
class RowDict(SqlTree):
    values: Dict[str, Sql]

    def _compile(self, qb):
        return {v.compile(qb).code + [f' as {name}'] for name, v in self.values.items()}


@dataclass
class Values(Table):
    type: Type
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile(qb) for v in self.values]
        if not values:  # SQL doesn't support empty values
            return ['SELECT NULL LIMIT 0']
        return ['VALUES '] + join_comma(parens(v.code) for v in values)


@dataclass
class AllFields(SqlTree):
    type: Type

    def _compile(self, qb):
        return ['*']

@dataclass
class Update(SqlTree):
    table: TableName
    fields: Dict[Sql, Sql]
    conds: List[Sql]
    type = T.null

    def _compile(self, qb):
        fields_sql = [k.compile(qb).code + [' = '] + v.compile(qb).code for k, v in self.fields.items()]
        fields_sql = join_comma(fields_sql)

        sql = ['UPDATE '] + self.table.compile(qb).code + [' SET '] + fields_sql

        if self.conds:
            sql += [' WHERE '] + join_sep([c.compile(qb).code for c in self.conds], ' AND ')

        return sql

@dataclass
class Delete(SqlTree):
    table: TableName
    conds: List[Sql]
    type = T.null

    def _compile(self, qb):
        conds = join_sep([c.compile(qb).code for c in self.conds], ' AND ')
        return ['DELETE FROM '] + self.table.compile(qb).code + [' WHERE '] + conds

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
        select_sql = join_comma(f.code for f in fields_sql)

        sql = ['SELECT '] + select_sql + [' FROM '] + self.table.compile(qb).code

        if self.conds:
            sql += [' WHERE '] + join_sep([c.compile(qb).code for c in self.conds], ' AND ')


        if self.group_by:
            sql += [' GROUP BY '] + join_comma(e.compile(qb).code for e in self.group_by)

        if self.limit:
            sql += [' LIMIT '] + self.limit.compile(qb).code
        elif self.offset:
            if qb.target == sqlite:
                sql += [' LIMIT -1']  # Sqlite only (and only old versions of it)

        if self.offset:
            sql += [' OFFSET '] + self.offset.compile(qb).code

        if self.order:
            sql += [' ORDER BY '] + join_comma(o.compile(qb).code for o in self.order)

        return sql


@listgen
def join_sep(code_list, sep):
    code_list = list(code_list)
    yield from code_list[0]
    for c in code_list[1:]:
        assert isinstance(c, list)
        yield sep
        yield from c

def join_comma(code_list):
    return join_sep(code_list, ", ")

@dataclass
class Subquery(SqlTree):
    table_name: str
    fields: List[Name]
    query: Sql
    type = property(X.query.type)

    def _compile(self, qb):
        query = self.query.compile(qb).code
        fields = [f.compile(qb.replace(is_root=False)).code for f in self.fields]
        fields_str = ["("] + join_comma(fields) + [")"] if fields else []
        return [f"{self.table_name}"] + fields_str + [" AS ("] + query + [")"]

    def compile(self, qb):
        sql_code = self._compile(qb)
        return CompiledSQL(self.type, sql_code, self, self._is_select, False)



@dataclass
class Join(TableOperation):
    type: Type
    join_op: str
    tables: List[Table]
    conds: List[Sql]

    def _compile(self, qb):
        tables_sql = [t.compile(qb).code for t in self.tables]
        join_op = ' %s ' % self.join_op.upper()
        join_sql = join_sep([e for e in tables_sql], join_op)

        if self.conds:
            conds = join_sep([c.compile(qb).code for c in self.conds], ' AND ')
        else:
            conds = ['1=1']   # Postgres requires ON clause

        return [f'SELECT * FROM '] + join_sql + [' ON '] + conds





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
class StringSlice(SqlTree):
    string: Sql
    start: Sql
    stop: Optional[Sql]

    type = T.string

    def _compile(self, qb):
        string = self.string.compile(qb).code
        start = self.start.compile(qb).code
        if self.stop:
            stop = self.stop.compile(qb).code
            length = parens(stop + ['-'] + start)
        else:
            length = None

        if qb.target == sqlite:
            f = 'substr'
            params = string + [', '] + start
            if length:
                params += [', '] + length
        elif qb.target == postgres:
            f = 'substring'
            params = string + [' from '] + start
            if length:
                params += [' for '] + length
        else:
            assert False

        return [f'{f}('] + params + [')']


@dp_type
def _repr(t: T.union[T.number, T.bool], x):
    return str(x)

@dp_type
def _repr(t: T.decimal, x):
    return repr(float(x))  # TODO SQL decimal?

@dp_type
def _repr(t: T.datetime, x):
    # TODO Better to pass the object instead of a string?
    return repr(str(x))

@dp_type
def _repr(t: T.union[T.string, T.text], x):
    return "'%s'" % str(x).replace("'", "''")

def make_value(x):
    if x is None:
        return null

    try:
        t = pql_types.from_python(type(x))
    except KeyError:
        raise ValueError(x)

    return Primitive(t, _repr(t, x))

def add_one(x):
    return Arith('+', [x, make_value(1)])


def compile_type_def(state, table_name, table) -> Sql:
    assert table <= T.table

    posts = []
    pks = []
    columns = []

    pks = {join_names(pk) for pk in table.options['pk']}
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
        if c <= T.t_relation:
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

@dp_type
def compile_type(type_: T.t_relation):
    # TODO might have a different type
    return 'INTEGER'    # Foreign-key is integer

@dp_type
def compile_type(type_: T.primitive):
    assert type_ <= T.primitive
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'text': "TEXT",
        't_relation': "INTEGER",
        'datetime': "TIMESTAMP",
    }[type_.typename]
    if not type_.nullable:
        s += " NOT NULL"
    return s

@dp_type
def compile_type(_type: T.null):
    return 'INTEGER'    # TODO is there a better value here? Maybe make it read-only somehow

@dp_type
def compile_type(idtype: T.t_id):
    s = "INTEGER"   # TODO non-int idtypes
    if not idtype.nullable:
        s += " NOT NULL"
    return s


