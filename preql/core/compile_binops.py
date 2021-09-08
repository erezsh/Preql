import operator
from preql import settings

from .pql_types import T, dp_inst
from .interp_common import pyvalue_inst, call_builtin_func 
from . import sql
from . import pql_objects as objects
from .exceptions import Signal
from .pql_objects import make_instance, remove_phantom_type
from .state import get_var
from .casts import cast

## Compare
def compare(op, a, b):
    res = _compare(op, remove_phantom_type(a), remove_phantom_type(b))
    return objects.inherit_phantom_type(res, [a, b])

@dp_inst
def _compare(op, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, op, f"Compare not implemented for {a.type} and {b.type}")


@dp_inst
def _compare(op, _a: T.nulltype, _b: T.nulltype):
    return pyvalue_inst(op in ('=', '<=', '>='))

@dp_inst
def _compare(_op, a: T.type, _b: T.nulltype):
    assert not a.type.maybe_null()
    return pyvalue_inst(False)
@dp_inst
def _compare(op, a: T.nulltype, b: T.type):
    return _compare(op, b, a)


primitive_or_struct = T.union[T.primitive, T.struct]

@dp_inst
def _compare(op, a: T.nulltype, b: primitive_or_struct):
    # TODO Enable this type-based optimization:
    # if not b.type.nullable:
    #     return objects.pyvalue_inst(False)
    if b.type <= T.struct:
        b = b.primary_key()
    code = sql.Compare(op, [a.code, b.code])
    return objects.Instance.make(code, T.bool, [a, b])
@dp_inst
def _compare(op, a: primitive_or_struct, b: T.nulltype):
    return _compare(op, b, a)


@dp_inst
def _compare(_op, _a: T.unknown, _b: T.object):
    return objects.UnknownInstance()
@dp_inst
def _compare(_op, _a: T.object, _b: T.unknown):
    return objects.UnknownInstance()
@dp_inst
def _compare(_op, _a: T.unknown, _b: T.unknown):
    return objects.UnknownInstance()


@dp_inst
def _prepare_to_compare(op, a, b):
    if op == '=':
        return pyvalue_inst(False)
    elif op == '!=':
        return pyvalue_inst(True)
    raise Signal.make(T.TypeError, op, f"Operator '{op}' not implemented for {a.type} and {b.type}")

@dp_inst
def _prepare_to_compare(op, a: T.number | T.bool, b: T.number | T.bool):
    pass
@dp_inst
def _prepare_to_compare(op, a: T.string, b: T.string):
    pass

# XXX id/relation can be either int or string, so we can't tell if comparison is necessary or not
# So we always allow the comparison 
# TODO use generics/phantoms to disambiguate the situation
id_or_relation = T.t_relation | T.t_id
@dp_inst
def _prepare_to_compare(op, a: id_or_relation, b):
    pass
@dp_inst
def _prepare_to_compare(op, a, b: id_or_relation):
    pass
@dp_inst
def _prepare_to_compare(op, a: id_or_relation, b: id_or_relation):
    pass

@dp_inst
def _compare(op, a: T.primitive, b: T.primitive):
    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        f = {
            '=': operator.eq,
            '!=': operator.ne,
            '<>': operator.ne,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
        }[op]
        try:
            return pyvalue_inst(f(a.local_value, b.local_value))
        except TypeError as e:
            raise Signal.make(T.TypeError, op, f"Operator '{op}' not implemented for {a.type} and {b.type}")

    # TODO regular equality for primitives? (not 'is')
    res = _prepare_to_compare(op, a, b)
    if res is not None:
        return res

    code = sql.Compare(op, [a.code, b.code])
    return objects.Instance.make(code, T.bool, [a, b])


@dp_inst
def _compare(op, a: T.type, b: T.type):
    if op == '<=':
        return call_builtin_func("issubclass", [a, b])
    if op != '=':
        raise Signal.make(T.NotImplementedError, op, f"Cannot compare types using: {op}")
    return pyvalue_inst(a == b)

@dp_inst
def _compare(op, a: T.primitive, b: T.row):
    return _compare(op, a, b.primary_key())

@dp_inst
def _compare(op, a: T.row, b: T.primitive):
    return _compare(op, b, a)

@dp_inst
def _compare(op, a: T.row, b: T.row):
    return _compare(op, a.primary_key(), b.primary_key())



## Contains
def contains(op, a, b):
    res = _contains(op, remove_phantom_type(a), remove_phantom_type(b))
    return objects.inherit_phantom_type(res, [a, b])


@dp_inst
def _contains(op, a: T.string, b: T.string):
    f = {
        'in': 'str_contains',
        '!in': 'str_notcontains',
    }[op]
    return call_builtin_func(f, [a, b])

@dp_inst
def _contains(op, a: T.primitive, b: T.table):
    b_list = cast(b, T.list)
    if not a.type <= b_list.type.elem:
        a = cast(a, b_list.type.elem)
        # raise Signal.make(T.TypeError, op, f"Error in contains: Mismatch between {a.type} and {b.type}")

    if op == '!in':
        op = 'not in'
    code = sql.Contains(op, [a.code, b_list.code])
    return objects.Instance.make(code, T.bool, [a, b_list])

@dp_inst
def _contains(op, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, op, f"Contains not implemented for {a.type} and {b.type}")



## Arith

def compile_arith(op, a, b):
    res = _compile_arith(op, remove_phantom_type(a), remove_phantom_type(b))
    return objects.inherit_phantom_type(res, [a, b])

@dp_inst
def _compile_arith(arith, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, arith.op, f"Operator '{arith.op}' not implemented for {a.type} and {b.type}")



@dp_inst
def _compile_arith(arith, a: T.table, b: T.table):
    # TODO validate types
    ops = {
        "+": 'table_concat',
        "&": 'table_intersect',
        "|": 'table_union',
        "-": 'table_subtract',
    }
    # TODO compile preql funccall?
    try:
        op = ops[arith.op]
    except KeyError:
        raise Signal.make(T.TypeError, arith.op, f"Operation '{arith.op}' not supported for tables ({a.type}, {b.type})")

    return get_var(op).func(a, b)



@dp_inst
def _compile_arith(arith, a: T.string, b: T.int):
    if arith.op != '*':
        raise Signal.make(T.TypeError, arith.op, f"Operator '{arith.op}' not supported between string and integer.")
    return call_builtin_func("repeat", [a, b])


@dp_inst
def _compile_arith(arith, a: T.number, b: T.number):
    if arith.op in ('/', '**') or a.type <= T.float or b.type <= T.float:
        res_type = T.float
    else:
        res_type = T.int

    try:
        f = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '/~': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
        }[arith.op]
    except KeyError:
        raise Signal.make(T.TypeError, arith, f"Operator {arith.op} not supported between types '{a.type}' and '{b.type}'")

    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        # Local folding for better performance.
        # However, acts a little different than SQL. For example, in this branch 1/0 raises ValueError,
        # while SQL returns NULL
        try:
            value = f(a.local_value, b.local_value)
        except ZeroDivisionError as e:
            raise Signal.make(T.ValueError, arith.args[-1], str(e))
        if arith.op == '**':
            value = float(value)
        return pyvalue_inst(value, res_type)

    code = sql.arith(res_type, arith.op, [a.code, b.code])
    return make_instance(code, res_type, [a, b])

@dp_inst
def _compile_arith(arith, a: T.string, b: T.string):
    if arith.op == 'like':
        code = sql.BinOp(T.bool, 'like', [a.code, b.code])
        return objects.Instance.make(code, T.bool, [a, b])

    if arith.op != '+':
        raise Signal.make(T.TypeError, arith.op, f"Operator '{arith.op}' not supported for strings.")

    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        # Local folding for better performance (optional, for better performance)
        return pyvalue_inst(a.local_value + b.local_value, T.string)

    code = sql.arith(T.string, arith.op, [a.code, b.code])
    return make_instance(code, T.string, [a, b])

