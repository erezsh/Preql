from copy import copy, deepcopy
from contextlib import suppress
import operator as ops

from . import ast_classes as ast
from . import pql_objects as pql
from .parser import parse, parse_expr
from .utils import dataclass, Dataclass, Context
from .sql import Sql
from . import sql
from .exceptions import PreqlError_MissingName



class State:
    def __init__(self):
        self.namespace = {}

class CompileSQL_Stmts:
    def __init__(self, state, interp):
        self.state = state
        self.interp = interp

    def to_sql(self, ast_node):
        ast_type = type(ast_node)
        assert issubclass(ast_type, ast.Stmt), ast_node
        return getattr(self, ast_type.__name__)(ast_node)

    def Column(self, column: ast.Column):
        mod = ' NOT NULL' if not column.is_nullable else ''
        mod += ' PRIMARY KEY' if column.is_pk else ''

        if isinstance(column.type, ast.RelationalType):
            c = f'{column.name} INTEGER{mod}'
            # TODO Add all foreign keys in the end! (can't be in middle of table)
            # fk = f'FOREIGN KEY({column.name}) REFERENCES {column.type.table_name}({column.type.column_name})'
            # return c + ", " + fk
            return c
        elif isinstance(column.type, ast.BackRefType):
            return None
        elif isinstance(column.type, ast.ManyToManyType):
            return None
        elif isinstance(column.type, ast.IdType):
            return f'{column.name} INTEGER{mod}'
        else:
            sql_type = {
                ast.IntegerType(): 'INTEGER',
                ast.StringType(): 'VARCHAR(4000)',
                ast.FloatType(): 'FLOAT'
                # 'real': 'REAL',
            }[column.type]
            return f'{column.name} {sql_type}{mod}'

    def TableDef(self, table: ast.TableDef):
        # TODO: If table exists, make sure schema is correct!!
        cols = [self.to_sql(r) for r in table.columns.values()]
        return ' '.join(['CREATE TABLE IF NOT EXISTS', table.name, '(\n ', ',\n  '.join(
                c for c in cols if c is not None
            ), '\n);'])

    def AddRow(self, addrow: ast.AddRow):
        cols = [c.name
                for c in self.state.namespace[addrow.table].columns.values()
                if not isinstance(c.type, (ast.BackRefType, ast.IdType))]

        values = EvalAst(self.state, self.interp)._eval_list(addrow.args)

        # # TODO verify types
        q = ['INSERT INTO', addrow.table,
             "(", ', '.join(cols), ")",
             "VALUES",
             "(", ', '.join(v.to_sql().compile().text for v in values), ")",
        ]
        insert = ' '.join(q) + ';'
        return insert



class EvalAst:
    def __init__(self, state, query_engine):
        self.state = state
        self.query_engine = query_engine
        self.context = Context()

        self._tables = {}

    def _eval(self, ast_node) -> pql.Object:
        ast_type = type(ast_node)
        assert issubclass(ast_type, ast.Expr), (ast_node)
        res = getattr(self, ast_type.__name__)(ast_node)
        assert isinstance(res, pql.Object), (ast_node, res)
        return res

    def _eval_list(self, l):
        return [self._eval(e) for e in l]

    def eval(self, ast_node, args = {}):
        with self.context.push(args=args):
            return self._eval(ast_node)

    def get_table(self, base_table, name):
        key = base_table, name
        if key not in self._tables:
            self._tables[key] = pql.JoinableTable(   self.state.namespace[name], self   )
        return self._tables[key]

    def Reference(self, ref: ast.Reference):
        if ref.name in self.state.namespace:
            obj = self.state.namespace[ref.name]
            if isinstance(obj, ast.TableDef):
                obj = pql.JoinableTable(obj, self)   # TODO generic interface?
        else:
            try:
                args = self.context.get('args')
                return args[ref.name]
            except KeyError:
                pass

            try:
                table = self.context.get('table')
            except KeyError:
                raise PreqlError_MissingName(ref.name)

            return table.getattr(ref.name)


        return obj

    def GetAttribute(self, getattr: ast.GetAttribute):
        obj = self._eval(getattr.obj)
        return obj.getattr(getattr.attr)

    def NewRow(self, newrow: ast.NewRow):
        table = self.state.namespace[newrow.table]
        assert not newrow.args.named_args, "Not supported yet"

        cols = [c.name
                for c in table.columns.values()
                if not isinstance(c.type, (ast.BackRefType, ast.IdType, ast.ManyToManyType))]

        values = self._eval_list(newrow.args.pos_args)

        assert len(cols) == len(values)

        # # TODO verify types

        insert = sql.Insert(table.name, cols, [v.to_sql() for v in values])
        assert not self.query_engine.query(insert, commit=True)

        rowid = self.query_engine.query(sql.LastRowId())
        return pql.RowRef(table, rowid, self.query_engine)



    def FuncCall(self, funccall: ast.FuncCall):
        func_obj = self._eval(funccall.obj)
        # assert isinstance(func_obj, Function)
        pos_args = funccall.args.pos_args
        named_args = funccall.args.named_args

        if isinstance(func_obj, pql.OrderTable):
            with self.context.push(table=func_obj.table):
                args = self._eval_list(pos_args)

            return func_obj.call(self.query_engine, args, named_args)
        elif isinstance(func_obj, pql.UserFunction):
            assert not named_args

            args = self._eval_list(pos_args)
            params = func_obj.funcdef.params or []
            assert len(args) == len(params) # TODO actual param resolution
            args = dict(zip(params, args))
            with self.context.push(args=args):
                return self._eval(func_obj.funcdef.expr)

        args = self._eval_list(pos_args)
        named_args = {name:self._eval(expr) for name,expr in named_args.items()}
        x = func_obj.call(self.query_engine, args, named_args)
        return x

    def Selection(self, sel: ast.Selection):
        obj = self._eval(sel.table)
        assert isinstance(obj, pql.Table)
        with self.context.push(table=obj):
            conds = self._eval_list(sel.conds)

        return pql.Query(obj, conds=conds)

    def Projection(self, proj: ast.Projection):
        obj = self._eval(proj.table)
        if isinstance(obj, pql.ColumnRef):
            assert isinstance(obj.type, ast.BackRefType)
            import pdb
            pdb.set_trace()

        assert isinstance(obj, pql.Table), obj
        with self.context.push(table=obj):
            fields = self._eval_list(proj.fields)
            agg_fields = self._eval_list(proj.agg_fields or [])

        return pql.Query(obj, fields=fields, agg_fields=agg_fields)

    def NamedExpr(self, ne: ast.NamedExpr): # XXX it's bad but i'm lazy
        expr = self._eval(ne.expr)
        if ne.name is None:
            return expr
        return pql.NamedExpr(ne.name, expr)

    def Contains(self, contains: ast.Contains):
        exprs = self._eval_list(contains.exprs)
        assert len(exprs) == 2
        return pql.Contains(contains.op, exprs)

    def Compare(self, cmp: ast.Compare):
        exprs = self._eval_list(cmp.exprs)
        assert len(exprs) == 2
        if all(isinstance(e, pql.Primitive) for e in exprs):
            op = {
                '<': ops.lt,
                '>': ops.gt,
                '<=': ops.le,
                '>=': ops.ge,
                '=': ops.eq,
                '!=': ops.ne,
                }[cmp.op]
            return pql.Bool(op(*[e.value for e in exprs]))

        return pql.Compare(cmp.op, exprs)

    # TODO const, probably parser can generate PreqlObject in the first place
    def Value(self, val: ast.Value):
        if isinstance(val.type, ast.IntegerType):
            return pql.Integer(val.value)
        if isinstance(val.type, ast.FloatType):
            return pql.Float(val.value)
        if isinstance(val.type, ast.StringType):
            return pql.String(val.value)
        if isinstance(val.type, ast.NullType):
            return pql.null
        raise NotImplementedError(val)

    def Arith(self, arith: ast.Arith):
        exprs = self._eval_list(arith.exprs)
        assert len(exprs) == 2
        if all(isinstance(e, pql.Primitive) for e in exprs):
            op = {
                '+': ops.add,
                '*': ops.mul,
                }[arith.op]
            return type(exprs[0])(op(*[e.value for e in exprs]))
        return pql.Arith(arith.op, exprs)

    def Neg(self, neg: ast.Neg):
        return pql.Neg(self._eval(neg.expr))

    def Desc(self, desc: ast.Desc):
        return pql.Desc(self._eval(desc.expr))


pql_functions = {
    'round': pql.Round,
    'count': pql.CountField,
    'limit': pql.LimitField,
    'join': pql.create_autojoin,
}



class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State()
        for name, func in pql_functions.items():
            self.state.namespace[name] = pql.SqlFunction(func)

        self.sqldecl = CompileSQL_Stmts(self.state, self)

    def _add_table(self, table):
        if table.name in self.state.namespace:
            raise Exception("Table already defined: %s" % table.name)
        self.state.namespace[table.name] = table

        backrefs = []

        for c in table.columns.values():
            if c.backref:
                if isinstance(c.type, ast.ManyToManyType):
                    ref_to = self.state.namespace[c.type.to_table.table_name]
                    backrefs.append((ref_to, ast.Column(c.backref, c.name, False, False, type=ast.ManyToManyType(ast.RelationalType(table.name)), table=table)))
                else:
                    assert isinstance(c.type, ast.RelationalType), c
                    ref_to = self.state.namespace[c.type.table_name]
                    backrefs.append((ref_to, ast.Column(c.backref, c.name, False, False, type=ast.BackRefType(table.name), table=table)))

        for ref_to, col in backrefs:
            assert col.name not in ref_to.columns
            ref_to.columns[col.name] = col

        r = self.sqlengine.query( self.sqldecl.to_sql(table) )
        assert not r

    def _add_row(self, addrow: ast.AddRow):
        insert = self.sqldecl.to_sql(addrow)
        assert not self.sqlengine.query(insert)

        if addrow.as_:
            rowid ,= self.sqlengine.query('SELECT last_insert_rowid();')[0]
            table = self.state.namespace[addrow.table]
            v = pql.RowRef(table, rowid, self)
            self.state.namespace[addrow.as_] = v

    def _def_function(self, func: ast.FunctionDef):
        assert func.name not in self.state.namespace
        self.state.namespace[func.name] = pql.UserFunction(func)

    def run_stmt(self, c):
        if isinstance(c, ast.TableDef):
            self._add_table(c)
        elif isinstance(c, ast.FunctionDef):
            self._def_function(c)
        elif isinstance(c, ast.AddRow):
            self._add_row(c)
        else:
            raise ValueError(c)

    def call_func(self, fname, args):
        args = [ast.Value.from_pyobj(a) for a in args]
        funccall = ast.FuncCall(ast.Reference(fname), ast.FuncArgs(args, {}))

        obj = EvalAst(self.state, self).eval(funccall)
        return obj

    def query(self, sql: Sql, commit=False):
        res = self._query_as_struct(sql.compile())
        if commit:
            self.sqlengine.commit()
        return res

    def eval_expr(self, code, args):
        expr_ast = parse_expr(code)
        obj = EvalAst(self.state, self).eval(expr_ast, args)

        return obj

    def _query_as_struct(self, compiled_sql):
        res = self.sqlengine.query(compiled_sql.text)
        return compiled_sql.type.import_value(res)

    def execute_code(self, code):
        for s in parse(code):
            self.run_stmt(s)


def _test(fn):
    a = open(fn).read()
    # a = open("preql/simple2.pql").read()
    # a = open("preql/tree.pql").read()
    from .api import SqliteEngine
    sqlengine = SqliteEngine()
    i = Interpreter(sqlengine)
    for s in parse(a):
        i.run_stmt(s)


def test():
    _test("preql/simple1.pql")
    # _test("preql/simple2.pql")
    # _test("preql/tree.pql")

if __name__ == '__main__':
    test()