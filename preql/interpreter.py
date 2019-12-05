from .utils import SafeDict

from .evaluate import State, execute, evaluate, simplify, localize
from .parser import parse_stmts, parse_expr
from . import pql_ast as ast
from . import pql_objects as objects
from . import pql_types as types

from .pql_functions import internal_funcs, joins


def initial_namespace():
    ns = SafeDict({p.name: p for p in types.primitives_by_pytype.values()})
    ns.update({
        name: objects.InternalFunction(name, [
            objects.Param(name) for name, type_ in list(f.__annotations__.items())[1:]
        ], f) for name, f in internal_funcs.items()
    })
    ns.update(joins)
    return [ns]

class Interpreter:
    def __init__(self, sqlengine):
        self.sqlengine = sqlengine
        self.state = State(sqlengine, 'text', initial_namespace())


    def call_func(self, fname, args):
        obj = simplify(self.state, ast.Name(fname))
        if isinstance(obj, objects.TableInstance):
            assert not args, args
            return localize(self.state, obj)

        funccall = ast.FuncCall(ast.Name(fname), args)
        return evaluate(self.state, funccall)

    def eval_expr(self, code, args):
        expr_ast = parse_expr(code)
        with self.state.use_scope(args):
            obj = evaluate(self.state, expr_ast)
        return obj

    def execute_code(self, code):
        for stmt in parse_stmts(code):
            try:
                execute(self.state, stmt)
            except:
                print("Error in statement: ", stmt)
                raise