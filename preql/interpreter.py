from .evaluate import State, execute, evaluate, simplify, localize
from .parser import parse_stmts, parse_expr
from . import pql_ast as ast
from . import pql_objects as objects

class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State(sqlengine, None)


    def call_func(self, fname, args):
        obj = simplify(self.state, ast.Name(fname))
        if isinstance(obj, objects.TableInstance):
            assert not args, args
            return localize(self.state, obj)

        funccall = ast.FuncCall(ast.Name(fname), args)
        return evaluate(self.state, funccall)

    def eval_expr(self, code, args):
        expr_ast = parse_expr(code)
        obj = evaluate(self.state, expr_ast)
        return obj

    def execute_code(self, code):
        for stmt in parse_stmts(code):
            try:
                execute(self.state, stmt)
            except:
                print("%%%", stmt)
                raise