from . import pql_objects as objects
from . import sql
from .pql_types import T, dp_type, ITEM_NAME
from .types_impl import kernel_type
from .exceptions import Signal
from .interp_common import call_builtin_func

@dp_type
def _cast(inst_type, target_type, inst):
    if inst_type <= target_type:
        return inst
    raise Signal.make(T.TypeError, None, f"Cast not implemented for {inst_type}->{target_type}")

@dp_type
def _cast(inst_type: T.list, target_type: T.list, inst):
    if inst is objects.EmptyList:
        return inst.replace(type=target_type)

    if inst_type.elem <= target_type.elem:
        return inst

    value = inst.get_column(ITEM_NAME)
    elem = _cast(value.type, target_type.elem, value)
    code = sql.Select(target_type, inst.code, [sql.ColumnAlias(elem.code, ITEM_NAME)])
    return inst.replace(code=code, type=T.list[elem.type])


@dp_type
def _cast(inst_type: T.aggregated, target_type: T.list, inst):
    res = _cast(inst_type.elem, target_type.elem, inst)
    return objects.aggregate(res)   # ??

@dp_type
def _cast(inst_type: T.table, target_type: T.list, inst):
    t = inst.type
    if len(t.elems) != 1:
        raise Signal.make(T.TypeError, None, f"Cannot cast {inst_type} to {target_type}. Too many columns")
    if not inst_type.elem <= target_type.elem:
        raise Signal.make(T.TypeError, None, f"Cannot cast {inst_type} to {target_type}. Elements not matching")

    (elem_name, elem_type) ,= inst_type.elems.items()
    code = sql.Select(T.list[elem_type], inst.code, [sql.ColumnAlias(sql.Name(elem_type, elem_name), ITEM_NAME)])

    return objects.TableInstance.make(code, T.list[elem_type], [inst])

@dp_type
def _cast(inst_type: T.table, target_type: T.primitive, inst):
    t = inst.type
    if len(t.elems) != 1:
        raise Signal.make(T.TypeError, None, f"Cannot cast {inst_type} to {target_type}. Expected exactly 1 column, instead got {len(t.elems)}")
    if not inst_type.elem <= target_type:
        raise Signal.make(T.TypeError, None, f"Cannot cast {inst_type} to {target_type}. Elements type doesn't match")

    res = inst.localize()
    if len(res) != 1:
        raise Signal.make(T.TypeError, None, f"Cannot cast {inst_type} to {target_type}. Expected exactly 1 row, instead got {len(res)}")
    item ,= res
    return objects.pyvalue_inst(item, inst_type.elem)
 

@dp_type
def _cast(_inst_type: T.t_id, _target_type: T.int, inst):
    return inst.replace(type=T.int)

@dp_type
def _cast(_inst_type: T.int, target_type: T.t_id, inst):
    return inst.replace(type=target_type)

@dp_type
def _cast(_inst_type: T.union[T.float, T.bool], _target_type: T.int, inst):
    code = sql.Cast(T.int, inst.code)
    return objects.Instance.make(code, T.int, [inst])

@dp_type
def _cast(_inst_type: T.number, _target_type: T.bool, inst):
    code = sql.Compare('!=', [inst.code, sql.make_value(0)])
    return objects.Instance.make(code, T.bool, [inst])

@dp_type
def _cast(_inst_type: T.string, _target_type: T.bool, inst):
    code = sql.Compare('!=', [inst.code, sql.make_value('')])
    return objects.Instance.make(code, T.bool, [inst])

@dp_type
def _cast(_inst_type: T.string, _target_type: T.text, inst):
    return inst.replace(type=T.text)
@dp_type
def _cast(_inst_type: T.text, _target_type: T.string, inst):
    return inst.replace(type=T.string)
@dp_type
def _cast(_inst_type: T.string, _target_type: T.string, inst):     # Disambiguate text<->string due to inheritance
    return inst

@dp_type
def _cast(_inst_type: T.union[T.int, T.bool], _target_type: T.float, inst):
    code = sql.Cast(T.float, inst.code)
    return objects.Instance.make(code, T.float, [inst])

@dp_type
def _cast(_inst_type: T.string, _target_type: T.int, inst):
    return call_builtin_func("_cast_string_to_int", [inst])


# @dp_type
# def _cast(_inst_type: T.string, _target_type: T.datetime, inst):
#     # XXX unsafe cast, bad strings won't throw an error
#     return objects.Instance.make(inst.code, T.datetime, [inst])

@dp_type
def _cast(_inst_type: T.primitive, _target_type: T.string, inst):
    code = sql.Cast(T.string, inst.code)
    return objects.Instance.make(code, T.string, [inst])

@dp_type
def _cast(_inst_type: T.t_relation, target_type: T.t_id, inst):
    # TODO verify same table? same type?
    return inst.replace(type=target_type)

@dp_type
def _cast(inst_type: T.t_relation, target_type: T.int, inst):
    if inst.type.elem <= T.int:
        return inst.replace(type=target_type)
    raise Signal.make(T.TypeError, None, f"Cast not implemented for {inst_type}->{target_type}")

def cast(obj, t):
    res = _cast(kernel_type(obj.type), t, obj)
    return objects.inherit_phantom_type(res, [obj])
