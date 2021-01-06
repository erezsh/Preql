import json
from datetime import datetime
from typing import List, Optional, Dict

from .utils import dataclass, X, listgen, field_list
from . import pql_types
from .pql_types import ITEM_NAME, T, Type, dp_type, dp_inst, Id
from .types_impl import join_names, flatten_type
from .exceptions import Signal

duck = 'duck'
sqlite = 'sqlite'
postgres = 'postgres'
bigquery = 'bigquery'
mysql = 'mysql'

class QueryBuilder:
    def __init__(self, target, is_root=True, start_count=0):
        self.target = target
        self.is_root = is_root

        self.counter = start_count

        self.table_name = []

    def unique_name(self):
        self.counter += 1
        return 't%d' % self.counter

    def replace(self, is_root):
        if is_root == self.is_root:
            return self # Optimize
        return QueryBuilder(self.target, is_root, self.counter)

    def push_table(self, t):
        self.table_name.append(t)
    def pop_table(self, t):
        t2 = self.table_name.pop()
        assert t2 == t

    def safe_name(self, base):
        "Return a name that is safe for use as variable. Must be consistent (pure func)"
        return _quote(self.target, base)

    def quote(self, id_):
        assert isinstance(id_, Id)
        return '.'.join(self.safe_name(n) for n in id_.parts)



class Sql:
    pass

@dataclass
class SqlTree(Sql):
    _is_select = False
    _needs_select = False

    _compile = NotImplemented

    def compile_wrap(self, qb):  # Move to Expr? Doesn't apply to statements
        return self.compile(qb).wrap(qb)

    def compile(self, qb):
        sql_code = self._compile(qb.replace(is_root=False))
        assert isinstance(sql_code, list), self
        assert all(isinstance(c, (str, Parameter)) for c in sql_code), self

        return CompiledSQL(self.type, sql_code, self, self._is_select, self._needs_select)

    def finalize_with_subqueries(self, qb, subqueries):
        if subqueries:
            subqs = [q.compile_wrap(qb).finalize(qb) for (name, q) in subqueries.items()]
            sql_code = ['WITH RECURSIVE ']
            sql_code += join_comma([q, '\n    '] for q in subqs)
        else:
            sql_code = []
        sql_code += self.compile_wrap(qb).finalize(qb)
        return ''.join(sql_code)


@dataclass
class CompiledSQL(Sql):
    type: Type
    code: list
    source_tree: Optional[Sql]
    _is_select: bool   # Needed for embedding in SqlTree
    _needs_select: bool

    def finalize(self, qb):
        wrapped = self.wrap(qb)
        assert qb.is_root
        if wrapped.type <= T.primitive and not wrapped.code[0].lower().startswith('select '):
            code = ['SELECT '] + wrapped.code
        else:
            code = wrapped.code
        return ''.join(code)

    def wrap(self, qb):
        code = self.code

        if qb.is_root:
            if self._needs_select:
                code = ['SELECT * FROM '] + code
                return self.replace(code=code, _needs_select=False, _is_select=True)
        else:
            if self._is_select and not self._needs_select:
                # Bad recursion
                if qb.target == 'postgres' or qb.target == 'mysql' or qb.target == 'duck':
                    code = ['('] + code + [') ', qb.unique_name()]  # postgres requires an alias
                else:
                    code = ['('] + code + [')']

                return self.replace(code=code, _is_select=False)

        return self

    def compile_wrap(self, qb):
        return self.wrap(qb)
    def compile(self, qb):
        return self
    def finalize_with_subqueries(self, qb, subqueries):
        return SqlTree.finalize_with_subqueries(self, qb, subqueries)

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
    type = T.nulltype

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
    name: Id

    def _compile(self, qb):
        return [qb.quote(self.name)]

    _needs_select = True


class TableOperation(Table):
    _is_select = True


@dataclass
class FieldFunc(SqlTree):
    name: str
    field: Sql
    type = T.int

    def _compile(self, qb):
        return [f'{self.name}('] + self.field.compile_wrap(qb).code + [')']


@dataclass
class CountTable(Scalar):
    table: Sql
    type = T.int

    def _compile(self, qb):
        return [f'(SELECT COUNT(*) FROM '] + self.table.compile_wrap(qb).code + [')']

@dataclass
class FuncCall(SqlTree):
    type: Type
    name: str
    fields: List[Sql]

    def _compile(self, qb):
        s = join_comma(f.compile_wrap(qb).code for f in self.fields)
        return [f'{self.name}('] + s + [')']

@dataclass
class Cast(SqlTree):
    type: Type
    as_type: str
    value: Sql

    def _compile(self, qb):
        return [f'CAST('] + self.value.compile_wrap(qb).code + [f' AS {self.as_type})']


@dataclass
class Case(SqlTree):
    "SQL conditional"
    cond: Sql
    then: Sql
    else_: Optional[Sql]

    type = T.bool

    def _compile(self, qb):
        cond = self.cond.compile_wrap(qb).code
        then = self.then.compile_wrap(qb).code
        code = ["CASE WHEN "] + cond +[" THEN "] + then
        if self.else_:
            code += [ " ELSE " ] + self.else_.compile_wrap(qb).code
        return code + [" END "]

_ARRAY_SEP = '||'

@dataclass
class MakeArray(SqlTree):
    type: Type
    field: Sql

    def _compile(self, qb):
        field = self.field.compile_wrap(qb).code
        if qb.target == sqlite:
            return ['group_concat('] + field + [f', "{_ARRAY_SEP}")']
        elif qb.target == postgres:
            return ['array_agg('] + field + [')']
        elif qb.target == mysql:
            return ['json_arrayagg('] + field + [')']

        assert False, qb.target


@dataclass
class Contains(Scalar):
    op: str
    exprs: List[Sql]
    type = T.bool

    def _compile(self, qb):
        assert self.op
        item, container = self.exprs
        c_item = item.compile_wrap(qb).code
        c_cont = container.compile_wrap(qb.replace(is_root=True)).code
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
        elems = [e.compile_wrap(qb).code for e in self.exprs]

        op = self.op

        if any(e.type.maybe_null() for e in self.exprs):
            # Null values are possible, so we'll use identity operators
            if qb.target == sqlite:
                op = {
                    '=': 'is',
                    '!=': 'is not'
                }.get(op, op)
            elif qb.target is mysql:
                if op == '!=':
                    # Special case,
                    return parens( ['not '] + join_sep(elems, f' <=> ') )

                op = {
                    '=': '<=>',
                }.get(op, op)
            else:
                op = {
                    '=': 'is not distinct from',
                    '!=': 'is distinct from'
                }.get(op, op)

        return parens( join_sep(elems, f' {op} ') )


@dataclass
class LogicalBinOp(Scalar):
    op: str
    exprs: List[Sql]

    def _compile(self, qb):
        x = join_sep([e.compile_wrap(qb).code for e in self.exprs], f' {self.op} ')
        return parens(x)

    type = T.bool

@dataclass
class LogicalNot(Scalar):
    expr: Sql

    def _compile(self, qb):
        x = ['NOT '] + self.expr.compile_wrap(qb).code
        return parens(x)

    type = T.bool


@dataclass
class BinOp(Scalar):
    op: str
    exprs: List[Sql]

    def _compile(self, qb):
        x = join_sep([e.compile_wrap(qb).code for e in self.exprs], f' {self.op} ')
        return parens(x)

    type = property(X.exprs[0].type)     # TODO ensure type correctness


@dataclass
class TableArith(TableOperation):
    op: str
    exprs: List[Table]

    def _compile(self, qb):
        tables = [t.compile_wrap(qb) for t in self.exprs]
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
        s = self.expr.compile_wrap(qb)
        return ["-"] + s.code

    type = property(X.expr.type)

@dataclass
class Desc(SqlTree):
    expr: Sql

    def _compile(self, qb):
        s = self.expr.compile_wrap(qb)
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
        value = self.value.compile_wrap(qb).code
        assert alias and value, (alias, value)
        if value == [alias]:  # TODO disable when unoptimized?
            return value  # This is just for beauty, it's not necessary for function

        return value + [f' AS {alias}']

    type = property(X.value.type)


@dataclass
class Insert(SqlTree):
    table_name: Id
    columns: List[str]
    query: Sql
    type = T.nulltype

    def _compile(self, qb):
        columns = [qb.quote(Id(c)) for c in self.columns]
        return [f'INSERT INTO {qb.quote(self.table_name)}({", ".join(columns)}) '] + self.query.compile(qb).code

    def finalize_with_subqueries(self, qb, subqueries):
        if qb.target is mysql:
            columns = [qb.quote(Id(c)) for c in self.columns]
            sql_code = f'INSERT INTO {qb.quote(self.table_name)}({", ".join(columns)}) '
            sql_code += self.query.finalize_with_subqueries(qb, subqueries)
            return ''.join(sql_code)

        return super().finalize_with_subqueries(qb, subqueries)

@dataclass
class InsertConsts(SqlTree):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.nulltype

    def _compile(self, qb):
        assert self.tuples, self

        values = join_comma(
            parens(join_comma([e.compile_wrap(qb).code for e in tpl]))
            for tpl in self.tuples
        )

        cols = [qb.quote(Id(c)) for c in self.cols]
        q = ['INSERT INTO', qb.safe_name(self.table),
             "(", ', '.join(cols), ")",
             "VALUES ",
        ]
        return [' '.join(q)] + values #+ [';']

@dataclass
class InsertConsts2(SqlTree):
    table: str
    cols: List[str]
    tuples: list #List[List[Sql]]
    type = T.nulltype

    def _compile(self, qb):
        assert self.tuples, self

        values = join_comma(
            parens(join_comma([t] for t in tpl))
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
        elif qb.target == mysql:
            return ['last_insert_id()']
        else:
            return ['lastval()']   # Postgres

@dataclass
class SelectValue(Atom, TableOperation):
    # XXX Just use a regular select?
    value: Sql

    def _compile(self, qb):
        value = self.value.compile_wrap(qb)
        return [f'SELECT '] + value.code + [' AS '] + value

    type = property(X.value.type)

@dataclass
class RowDict(SqlTree):
    values: Dict[str, Sql]

    def _compile(self, qb):
        return {v.compile_wrap(qb).code + [f' as {name}'] for name, v in self.values.items()}


@dataclass
class Values(Table):
    type: Type
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile_wrap(qb) for v in self.values]
        if not values:  # SQL doesn't support empty values
            nulls = ', '.join(['NULL' for _ in range(len(self.type.elems))])
            return ['SELECT ' + nulls + ' LIMIT 0']

        if qb.target == mysql:
            def row_func(x):
                return ['ROW('] + x + [')']
        else:
            row_func = parens
        return ['VALUES '] + join_comma(row_func(v.code) for v in values)

@dataclass
class Tuple(SqlTree):
    type: Type
    values: List[Sql]

    def _compile(self, qb):
        values = [v.compile_wrap(qb).code for v in self.values]
        return join_comma(values)

@dataclass
class ValuesTuples(Table):
    type: Type
    values: List[Tuple]

    def _compile(self, qb):
        if not self.values:  # SQL doesn't support empty values
            return ['SELECT '] + join_comma(['NULL']*len(self.type.elems)) +  ['LIMIT 0']
        values = [v.compile_wrap(qb) for v in self.values]
        return ['VALUES '] + join_comma(v.code for v in values)



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
    type = T.nulltype

    def _compile(self, qb):
        fields_sql = [k.compile_wrap(qb).code + [' = '] + v.compile_wrap(qb).code for k, v in self.fields.items()]
        fields_sql = join_comma(fields_sql)

        sql = ['UPDATE '] + self.table.compile_wrap(qb).code + [' SET '] + fields_sql

        if self.conds:
            sql += [' WHERE '] + join_sep([c.compile_wrap(qb).code for c in self.conds], ' AND ')

        return sql

@dataclass
class Delete(SqlTree):
    table: TableName
    conds: List[Sql]
    type = T.nulltype

    def _compile(self, qb):
        conds = join_sep([c.compile_wrap(qb).code for c in self.conds], ' AND ')
        return ['DELETE FROM '] + self.table.compile_wrap(qb).code + [' WHERE '] + conds

@dataclass
class Select(TableOperation):
    type: Type
    table: Sql # XXX Table won't work with RawSQL
    fields: List[Sql]
    conds: List[Sql] = field_list()
    group_by: List[Sql] = field_list()
    order: List[Sql] = field_list()

    # MySQL doesn't support arithmetic in offset/limit, and we don't need it anyway
    offset: Optional[int] = None
    limit: Optional[int] = None

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

            # elif s1._is_limit_only() and not (s2.offset or s2.limit):
            #     s = s2.replace(limit=s1.limit, offset=s1.offset)
            #     return s._compile(qb)

        #
        # Compile
        #
        fields_sql = [f.compile_wrap(qb) for f in self.fields]
        select_sql = join_comma(f.code for f in fields_sql)

        sql = ['SELECT '] + select_sql + [' FROM '] + self.table.compile_wrap(qb).code

        if self.conds:
            sql += [' WHERE '] + join_sep([c.compile_wrap(qb).code for c in self.conds], ' AND ')


        if self.group_by:
            sql += [' GROUP BY '] + join_comma(e.compile_wrap(qb).code for e in self.group_by)

        if self.order:
            sql += [' ORDER BY '] + join_comma(o.compile_wrap(qb).code for o in self.order)

        if self.limit is not None:
            sql += [' LIMIT ', str(self.limit)]
        elif self.offset is not None:
            if qb.target == sqlite:
                sql += [' LIMIT -1']  # Sqlite only (and only old versions of it)
            elif qb.target == mysql:
                # MySQL requires a specific limit, always!
                # See: https://stackoverflow.com/questions/255517/mysql-offset-infinite-rows
                sql += [' LIMIT 18446744073709551615']

        if self.offset is not None:
            sql += [' OFFSET ', str(self.offset)]

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
        fields = [f.compile_wrap(qb.replace(is_root=False)).code for f in self.fields]
        fields_str = ["("] + join_comma(fields) + [")"] if fields else []
        return [f"{self.table_name}"] + fields_str + [" AS ("] + query + [")"]

def _enum_is_last(seq):
    last = len(seq) - 1
    for i, item in enumerate(seq):
        yield i == last, item


@dataclass
class Join(TableOperation):
    type: Type
    join_op: str
    tables: List[Table]
    conds: List[Sql]

    def _compile(self, qb):
        tables_sql = [t.compile_wrap(qb).code for t in self.tables]
        join_op = ' %s ' % self.join_op.upper()

        code = [f'SELECT * FROM '] + tables_sql[0]

        for is_last, t_sql in _enum_is_last(tables_sql[1:]):
            code += [join_op] + t_sql + [' ON ']

            if self.conds and is_last:
                code += join_sep([c.compile_wrap(qb).code for c in self.conds], ' AND ')
            else:
                code += ['1=1']   # Postgres requires ON clause

        return code





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
    fields = [Name(list_type.elem, ITEM_NAME)]
    subq = Subquery(name, fields, Values(list_type, elems))
    table = TableName(list_type, Id(name))
    return table, subq

def create_table(table_type, name, rows):
    fields = [Name(col_type, col_name) for col_name, col_type in table_type.elems.items()]
    subq = Subquery(name, fields, Values(table_type, rows))
    table = TableName(table_type, Id(name))
    return table, subq

def table_slice(table, start, stop):
    limit = stop - start if stop else None
    return Select(table.type, table.code, [AllFields(table.type)], offset=start, limit=limit)

def table_selection(table, conds):
    return Select(table.type, table.code, [AllFields(table.type)], conds=conds)

def table_order(table, fields):
    return Select(table.type, table.code, [AllFields(table.type)], order=fields)

def arith(target, res_type, op, args):
    arg_codes = list(args)
    if res_type == T.string:
        assert op == '+'
        op = '||'
        if target is mysql: # doesn't support a || b
            return FuncCall(res_type, 'concat', arg_codes)
    elif op == '/':
        if target != mysql:
            # In MySQL division returns a float. All others return int
            arg_codes[0] = Cast(T.float, 'float', arg_codes[0])
    elif op == '/~':
        if target == mysql:
            op = 'DIV'
        else:
            op = '/'

    return BinOp(op, arg_codes)


@dataclass
class StringSlice(SqlTree):
    string: Sql
    start: Sql
    stop: Optional[Sql]

    type = T.string

    def _compile(self, qb):
        string = self.string.compile_wrap(qb).code
        start = self.start.compile_wrap(qb).code
        if self.stop:
            stop = self.stop.compile_wrap(qb).code
            length = parens(stop + ['-'] + start)
        else:
            length = None

        if qb.target == sqlite:
            f = 'substr'
            params = string + [', '] + start
            if length:
                params += [', '] + length
        elif qb.target in (postgres, mysql):
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
    return BinOp('+', [x, make_value(1)])

def _quote(target, name):
    assert isinstance(name, str), name
    if target is sqlite:
        return f'[{name}]'
    elif target is mysql or target is bigquery:
        return f'`{name}`'
    else:
        return f'"{name}"'

def compile_type_def(state, table_name, table) -> Sql:
    assert table <= T.table

    target = state.db.target

    posts = []
    pks = []
    columns = []

    pks = {join_names(pk) for pk in table.options['pk']}
    for name, c in flatten_type(table):
        if name in pks and c <= T.t_id:
            if target == postgres:
                type_ = "SERIAL" # Postgres
            elif target == mysql:
                type_ = "INT NOT NULL AUTO_INCREMENT"
            else:
                type_ = "INTEGER"   # TODO non-int idtypes
        else:
            type_ = compile_type(c)

        columns.append( f'{_quote(target, name)} {type_}' )

        if c <= T.t_relation:
            # TODO any column, using projection / get_attr
            if not table.options.get('temporary', False):
                # In postgres, constraints on temporary tables may reference only temporary tables
                rel = c.options['rel']
                if rel['key']:          # Requires a unique constraint
                    _tbl_name ,= rel['table'].options['name'].parts   # TODO fix for multiple parts
                    s = f"FOREIGN KEY({name}) REFERENCES {_quote(target, _tbl_name)}({rel['column']})"
                    posts.append(s)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    command = "CREATE TEMPORARY TABLE" if table.options.get('temporary', False) else "CREATE TABLE IF NOT EXISTS"
    return RawSql(T.nulltype, f'{command} {_quote(target, table_name)} (' + ', '.join(columns + posts) + ')')

def compile_drop_table(state, table_name) -> Sql:
    target = state.db.target
    return RawSql(T.nulltype, f'DROP TABLE {_quote(target, table_name)}')

@dp_type
def compile_type(type_: T.t_relation):
    # TODO might have a different type
    #return 'INTEGER'    # Foreign-key is integer
    return compile_type(type_.elems['item'])

@dp_type
def compile_type(type_: T.primitive):
    assert type_ <= T.primitive
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'text': "TEXT",
        # 't_relation': "INTEGER",
        'datetime': "TIMESTAMP",
    }[type_.typename]
    if not type_.maybe_null():
        s += " NOT NULL"
    return s

@dp_type
def compile_type(_type: T.nulltype):
    return 'INTEGER'    # TODO is there a better value here? Maybe make it read-only somehow

@dp_type
def compile_type(idtype: T.t_id):
    s = "INTEGER"   # TODO non-int idtypes
    if not idtype.maybe_null():
        s += " NOT NULL"
    return s




def _from_datetime(state, s):
    if s is None:
        return None

    # Postgres
    if isinstance(s, datetime):
        return s

    # Sqlite
    if not isinstance(s, str):
        raise Signal.make(T.TypeError, [], None, f"datetime expected a string. Instead got: {s}")
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise Signal.make(T.ValueError, state, None, str(e))


@dp_inst
def from_sql(state, res: T.primitive):
    try:
        row ,= res.value
        item ,= row
    except ValueError:
        raise Signal.make(T.TypeError, state, None, "Expected primitive. Got: '%s'" % res.value)
    # t = from_python(type(item))
    # if not (t <= res.type):
    #     raise Signal.make(T.TypeError, state, None, f"Incorrect type returned from SQL: '{t}' instead of '{res.type}'")
    return item


@dp_inst
def from_sql(state, res: T.datetime):
    # XXX doesn't belong here?
    row ,= res.value
    item ,= row
    s = item
    return _from_datetime(state, s)

@dp_inst
def from_sql(state, arr: T.list):
    if not all(len(e)==1 for e in arr.value):
        raise Signal.make(T.TypeError, state, None, f"Expected 1 column. Got {len(arr.value[0])}")
    return [e[0] for e in arr.value]

@dp_inst
@listgen
def from_sql(state, arr: T.table):
    expected_length = len(flatten_type(arr.type))   # TODO optimize?
    for row in arr.value:
        if len(row) != expected_length:
            raise Signal.make(T.TypeError, state, None, f"Expected {expected_length} columns, but got {len(row)}")
        i = iter(row)
        yield {name: restructure_result(state, col, i) for name, col in arr.type.elems.items()}

@dp_type
def restructure_result(state, t: T.table, i):
    # return ({name: restructure_result(state, col, i) for name, col in t.elem_dict.items()})
    return next(i)

@dp_type
def restructure_result(state, t: T.struct, i):
    return ({name: restructure_result(state, col, i) for name, col in t.elems.items()})

@dp_type
def restructure_result(state, t: T.union[T.primitive, T.nulltype], i):
    return next(i)

@dp_type
def restructure_result(state, t: T.vectorized[T.union[T.primitive, T.nulltype]], i):
    return next(i)


@dp_type
def restructure_result(state, t: T.list[T.union[T.primitive, T.nulltype]], i):
    res = next(i)
    if state.db.target == mysql:
        res = json.loads(res)
    elif state.db.target == sqlite:
        res = res.split(_ARRAY_SEP)

    # XXX hack! TODO Use a generic form to cast types
    if t.elem <= T.int:
        res = [int(x) for x in res]
    elif t.elem <= T.float:
        res = [float(x) for x in res]
    return res

@dp_type
def restructure_result(state, t: T.datetime, i):
    s = next(i)
    return _from_datetime(None, s)
