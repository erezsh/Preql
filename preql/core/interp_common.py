from logging import getLogger

from preql.utils import dsp
from preql.context import context

from . import pql_ast as ast
from . import pql_objects as objects
from .exceptions import Signal
from .pql_types import Type, T


logger = getLogger('interp')

# Define common dispatch functions

@dsp
def evaluate( obj: type(NotImplemented)) -> object:
    raise NotImplementedError()

@dsp
def cast_to_python(obj: type(NotImplemented)) -> object:
    raise NotImplementedError(obj)



def assert_type(t, type_, ast_node, op, msg="%s expected an object of type %s, instead got '%s'"):
    assert isinstance(t, Type), t
    assert isinstance(type_, Type)
    if not t <= type_:
        if type_.typename == 'union':
            type_str = ' or '.join("'%s'" % elem for elem in type_.elems)
        else:
            type_str = "'%s'" % type_
        raise Signal.make(T.TypeError, ast_node, msg % (op, type_str, t))

def exclude_fields(table, fields):
    proj = ast.Projection(table, [ast.NamedField(None, ast.Ellipsis(None, exclude=list(fields) ), user_defined=False)])
    return evaluate(proj)

def call_builtin_func(name, args):
    "Call a builtin pql function"
    builtins = context.state.ns.get_var('__builtins__')
    assert isinstance(builtins, objects.Module)

    expr = ast.FuncCall(builtins.namespace[name], args)
    return evaluate( expr)



def is_global_scope(state):
    assert len(state.ns) != 0
    return len(state.ns) == 1


# def cast_to_python_primitive(state, obj):
#     res = cast_to_python(obj)
#     assert isinstance(res, (int, str, float, dict, list, type(None), datetime)), (res, type(res))
#     return res

def cast_to_python_string(obj: objects.AbsInstance):
    res = cast_to_python(obj)
    if not isinstance(res, str):
        raise Signal.make(T.TypeError, obj, f"Expected string, got '{res}'")
    return res

def cast_to_python_int(obj: objects.AbsInstance):
    res = cast_to_python(obj)
    if not isinstance(res, int):
        raise Signal.make(T.TypeError, obj, f"Expected int, got '{res}'")
    return res

    
pyvalue_inst = objects.pyvalue_inst
