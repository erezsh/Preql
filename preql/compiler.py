import re
import operator

from runtype import DispatchError

from .utils import safezip, listgen, find_duplicate, dataclass, SafeDict, re_split
from .exceptions import Signal
from . import exceptions as exc

from . import settings
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import dy, State, assert_type, new_value_instance, evaluate, simplify, call_pql_func, cast_to_python
from .pql_types import T, Object, Type, union_types, Id, ITEM_NAME
from .types_impl import dp_inst, flatten_type, pql_repr
from .casts import cast
from .pql_objects import AbsInstance, vectorized, unvectorized, make_instance

class AutocompleteSuggestions(Exception):
    pass

@dy
def cast_to_instance(state, x: list):
    return [cast_to_instance(state, i) for i in x]

@dy
def cast_to_instance(state, x):
    try:
        x = simplify(state, x)  # just compile Name?
        inst = compile_to_inst(state, x)
        # inst = evaluate(state, x)
    except exc.ReturnSignal:
        raise Signal.make(T.CompileError, state, None, f"Bad compilation of {x}")

    if isinstance(inst, ast.ParameterizedSqlCode):
        raise exc.InsufficientAccessLevel(inst)

    if not isinstance(inst, AbsInstance):
        # TODO compile error? cast error?
        # TODO need to be able to catch this above, and provide better errors
        raise Signal.make(T.TypeError, state, None, f"Could not compile {pql_repr(state, inst.type, inst)}")

    return inst



@listgen
def _process_fields(state: State, fields):
    for f in fields:
        try:
            v = cast_to_instance(state, f.value)
        except Signal as e:
            if e.type <= T.TypeError:
                raise e.replace(message=f"Cannot use object of type '{evaluate(state, f.value).type}' in projection.")
            raise

        # TODO proper error message
        # if not isinstance(v, objects.AbsInstance):
        #     raise Signal.make(T.TypeError, state, None, f"Projection field is not an instance. Instead it is: {v}")

        # In Preql, {=>v} creates an array. In SQL, it selects the first element.
        # Here we mitigate that disparity.

        if v.type <= T.aggregate:
            v = v.primary_key()
            t = T.list[v.type]
            v = make_instance(sql.MakeArray(t, v.code), t, [v])

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name

        yield [name, v]


@listgen
def _expand_ellipsis(state, table, fields):
    direct_names = {f.value.name for f in fields if isinstance(f.value, ast.Name)}

    for f in fields:
        assert isinstance(f, ast.NamedField)

        if isinstance(f.value, ast.Ellipsis):
            if f.name:
                raise Signal.make(T.SyntaxError, state, f, "Cannot use a name for ellipsis (inlining operation doesn't accept a name)")
            else:
                t = table.type
                if t <= T.vectorized:
                    # FIXME why is this needed here? probably shouldn't be
                    elems = t.elem.elems
                else:
                    elems = t.elems

                for n in f.value.exclude:
                    if isinstance(n, ast.Marker):
                        raise AutocompleteSuggestions({k:(0, v) for k, v in elems.items()
                                                      if k not in direct_names
                                                      and k not in f.value.exclude})

                    if n not in elems:
                        raise Signal.make(T.NameError, state, n, f"Field to exclude '{n}' not found")
                    if n in direct_names:
                        raise Signal.make(T.NameError, state, n, f"Field to exclude '{n}' is explicitely included in projection")

                exclude = direct_names | set(f.value.exclude)

                for name in elems:
                    assert isinstance(name, str)
                    if name not in exclude:
                        yield ast.NamedField(name, ast.Name(name)).set_text_ref(f.text_ref)
        else:
            yield f


@dy
def compile_to_inst(state: State, x):
    return x
@dy
def compile_to_inst(state: State, node: ast.Ast):
    return node


@dy
def compile_to_inst(state: State, cb: ast.CodeBlock):
    if len(cb.statements) == 1:
        return compile_to_inst(state, cb.statements[0])

    # TODO some statements can be evaluated at compile time
    raise Signal.make(T.CompileError, state, cb, "Cannot compile this code block")
@dy
def compile_to_inst(state: State, i: ast.If):
    cond = cast(state, cast_to_instance(state, i.cond), T.bool)
    then = cast_to_instance(state, i.then)
    else_ = cast_to_instance(state, i.else_)
    code = sql.Case(cond.code, then.code, else_.code)
    return make_instance(code, T.bool, [cond, then, else_])


@dy
def compile_to_inst(state: State, proj: ast.Projection):
    table = cast_to_instance(state, proj.table)

    if table is objects.EmptyList:
        return table   # Empty list projection is always an empty list.

    t = T.union[T.collection, T.struct]
    if not (table.type <= T.union[t, T.vectorized[t]]):
        raise Signal.make(T.TypeError, state, proj, f"Cannot project objects of type {table.type}")

    fields = _expand_ellipsis(state, table, proj.fields)

    # Test duplicates in field names. If an automatic name is used, collision should be impossible
    dup = find_duplicate([f for f in list(proj.fields) + list(proj.agg_fields) if f.name], key=lambda f: f.name)
    if dup:
        raise Signal.make(T.TypeError, state, dup, f"Field '{dup.name}' was already used in this projection")

    attrs = table.all_attrs()

    with state.use_scope({n:vectorized(c) for n, c in attrs.items()}):
        fields = _process_fields(state, fields)

    for name, f in fields:
        t = T.union[T.primitive, T.struct, T.nulltype, T.unknown]
        if not (f.type <= T.union[t, T.vectorized[t]]):
            raise exc.Signal.make(T.TypeError, state, proj, f"Cannot project values of type: {f.type}")

    if isinstance(table, objects.StructInstance):
        t = T.struct({n:c.type for n, c in fields})
        return objects.StructInstance(t, dict(fields))

    agg_fields = []
    if proj.agg_fields:
        with state.use_scope({n:objects.aggregate(c) for n, c in attrs.items()}):
            agg_fields = _process_fields(state, proj.agg_fields)

    all_fields = fields + agg_fields

    # Make new type
    elems = {}
    # codename = state.unique_name('proj')
    for name_, inst in all_fields:
        assert isinstance(inst, AbsInstance)

        # TODO what happens if automatic name precedes and collides with user-given name?
        name = name_
        i = 1
        while name in elems:
            name = name_ + str(i)
            i += 1
        t = inst.type
        # Unvectorize (XXX is this correct?)
        if t <= T.vectorized:
            t = t.elem
        elems[name] = t

    # TODO inherit primary key? indexes?
    new_table_type = T.table(elems, temporary=False)    # XXX abstract=True

    # Make code
    flat_codes = [code
                  for _, inst in all_fields
                  for code in inst.flatten_code()]

    # TODO if nn != on
    sql_fields = [
        sql.ColumnAlias.make(code, nn)
        for code, (nn, _nt) in safezip(flat_codes, flatten_type(new_table_type))
    ]

    if not sql_fields:
        raise Signal.make(T.TypeError, state, proj, "No column provided for projection (empty projection)")

    # Make Instance
    new_table = objects.TableInstance.make(sql.null, new_table_type, [table] + [inst for _, inst in all_fields])

    groupby = []
    limit = None
    if proj.groupby:
        if fields:
            # groupby = [new_table.get_column(n).primary_key().code for n, rc in fields]
            groupby = [sql.Primitive(T.int, str(i+1)) for i in range(len(fields))]
        else:
            limit = 1
            # Alternatively we could
            #   groupby = [sql.null]
            # But postgres doesn't support it

    code = sql.Select(new_table_type, table.code, sql_fields, group_by=groupby, limit=limit)

    # Make Instance
    return new_table.replace(code=code)

@dy
def compile_to_inst(state: State, order: ast.Order):
    table = cast_to_instance(state, order.table)
    assert_type(table.type, T.table, state, order, "'order'")

    with state.use_scope(table.all_attrs()):
        fields = cast_to_instance(state, order.fields)

    for f in fields:
        if not f.type <= T.primitive:
            # TODO Support 'ordering' trait?
            raise Signal.make(T.TypeError, state, order, f"Arguments to 'order' must be primitive")


    code = sql.table_order(table, [c.code for c in fields])

    return objects.TableInstance.make(code, table.type, [table] + fields)

@dy
def compile_to_inst(state: State, expr: ast.DescOrder):
    obj = cast_to_instance(state, expr.value)
    return obj.replace(code=sql.Desc(obj.code))



@dy
def compile_to_inst(state: State, lst: list):
    return [evaluate(state, e) for e in lst]


def base_type(t):
    if t <= T.vectorized:
        return t.elem
    return t

@dy
def compile_to_inst(state: State, o: ast.Or):
    args = cast_to_instance(state, o.args)
    a, b = args
    if base_type(a.type) != base_type(b.type):
        raise Signal.make(T.TypeError, state, o, f"'or' operator requires both arguments to be of the same type, got {a.type} and {b.type}")
    cond = cast(state, a, T.bool)
    code = sql.Case(cond.code, a.code, b.code)
    return objects.make_instance(code, a.type, args)

@dy
def compile_to_inst(state: State, o: ast.And):
    args = cast_to_instance(state, o.args)
    a, b = args
    if base_type(a.type) != base_type(b.type):
        raise Signal.make(T.TypeError, state, o, f"'and' operator requires both arguments to be of the same type, got {a.type} and {b.type}")
    cond = cast(state, a, T.bool)
    code = sql.Case(cond.code, b.code, a.code)
    return objects.make_instance(code, a.type, args)

@dy
def compile_to_inst(state: State, o: ast.Not):
    expr = cast_to_instance(state, o.expr)
    expr_bool = cast(state, expr, T.bool)
    code = sql.LogicalNot(expr_bool.code)
    return objects.make_instance(code, T.bool, [expr])


@dy
def compile_to_inst(state: State, like: ast.Like):
    # XXX move to ast.Arith ?
    s = cast_to_instance(state, like.str)
    p = cast_to_instance(state, like.pattern)

    try:
        return _compile_arith(state, like, s, p)
    except DispatchError as e:
        raise Signal.make(T.TypeError, state, like, f"Like not implemented for {s.type} and {p.type}")


## Contains
@dp_inst
def _contains(state, op, a: T.string, b: T.string):
    f = {
        'in': 'str_contains',
        '!in': 'str_notcontains',
    }[op]
    return call_pql_func(state, f, [a, b])

@dp_inst
def _contains(state, op, a: T.primitive, b: T.collection):
    b_list = cast(state, b, T.list)
    if not (a.type <= b_list.type.elem):
        a = cast(state, a, b_list.type.elem)
        # raise Signal.make(T.TypeError, state, op, f"Error in contains: Mismatch between {a.type} and {b.type}")

    if op == '!in':
        op = 'not in'
    code = sql.Contains(op, [a.code, b_list.code])
    return objects.Instance.make(code, T.bool, [a, b_list])

@dp_inst
def _contains(state, op, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, state, op, f"Contains not implemented for {a.type} and {b.type}")

@dp_inst
def _contains(state, op, a: T.vectorized, b: T.any):
    return vectorized(_contains(state, op, unvectorized(a), b))

## Compare
@dp_inst
def _compare(state, op, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, state, op, f"Compare not implemented for {a.type} and {b.type}")

@dp_inst
def _compare(state, op, a: T.vectorized, b: T.vectorized):
    return vectorized(_compare(state, op, unvectorized(a), unvectorized(b)))

@dp_inst
def _compare(state, op, a: T.vectorized, b: T.any):
    return vectorized(_compare(state, op, unvectorized(a), b))

@dp_inst
def _compare(state, op, a: T.any, b: T.vectorized):
    return vectorized(_compare(state, op, a, unvectorized(b)))



@dp_inst
def _compare(state, op, a: T.nulltype, b: T.nulltype):
    return new_value_instance(op in ('=', '<=', '>='))

@dp_inst
def _compare(state, op, a: T.type, b: T.nulltype):
    assert not a.type.maybe_null()
    return new_value_instance(False)
@dp_inst
def _compare(state, op, a: T.nulltype, b: T.type):
    return _compare(state, op, b, a)


primitive_or_struct = T.union[T.primitive, T.struct]

@dp_inst
def _compare(state, op, a: T.nulltype, b: primitive_or_struct):
    # TODO Enable this type-based optimization:
    # if not b.type.nullable:
    #     return objects.new_value_instance(False)
    if b.type <= T.struct:
        b = b.primary_key()
    code = sql.Compare(op, [a.code, b.code])
    return objects.Instance.make(code, T.bool, [a, b])
@dp_inst
def _compare(state, op, a: primitive_or_struct, b: T.nulltype):
    return _compare(state, op, b, a)


@dp_inst
def _compare(state, op, a: T.unknown, b: T.object):
    return objects.UnknownInstance()
@dp_inst
def _compare(state, op, a: T.object, b: T.unknown):
    return objects.UnknownInstance()


@dp_inst
def _compare(state, op, a: T.primitive, b: T.primitive):
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
                return new_value_instance(f(a.local_value, b.local_value))

    # TODO regular equality for primitives? (not 'is')
    code = sql.Compare(op, [a.code, b.code])
    return objects.Instance.make(code, T.bool, [a, b])

@dp_inst
def _compare(state, ast_node, a: T.aggregate, b: T.aggregate):
    res = _compare(state, ast_node, a.elem, b.elem)
    return objects.aggregate(res)

@dp_inst
def _compare(state, ast_node, a: T.aggregate, b: T.int):
    res = _compare(state, ast_node, a.elem, b)
    return objects.aggregate(res)

@dp_inst
def _compare(state, ast_node, a: T.int, b: T.aggregate):
    return _compare(state, ast_node, b, a)

@dp_inst
def _compare(state, op, a: T.type, b: T.type):
    if op == '<=':
        return call_pql_func(state, "issubclass", [a, b])
    if op != '=':
        raise exc.Signal.make(T.NotImplementedError, state, op, f"Cannot compare types using: {op}")
    return new_value_instance(a == b)

@dp_inst
def _compare(state, op, a: T.number, b: T.row):
    return _compare(state, op, a, b.primary_key())

@dp_inst
def _compare(state, op, a: T.row, b: T.number):
    return _compare(state, op, b, a)

@dp_inst
def _compare(state, op, a: T.row, b: T.row):
    return _compare(state, op, a.primary_key(), b.primary_key())



@dy
def compile_to_inst(state: State, cmp: ast.Compare):
    insts = evaluate(state, cmp.args)

    if cmp.op == 'in' or cmp.op == '!in':
        return _contains(state, cmp.op, insts[0], insts[1])
    else:
        op = {
            '==': '=',
            '<>': '!=',
        }.get(cmp.op, cmp.op)
        return _compare(state, op, insts[0], insts[1])

@dy
def compile_to_inst(state: State, neg: ast.Neg):
    expr = cast_to_instance(state, neg.expr)
    assert_type(expr.type, T.number, state, neg, "Negation")

    return make_instance(sql.Neg(expr.code), expr.type, [expr])


@dy
def compile_to_inst(state: State, arith: ast.Arith):
    args = evaluate(state, arith.args)

    try:
        return _compile_arith(state, arith, *args)
    except DispatchError as e:
        a, b = args
        raise Signal.make(T.TypeError, state, arith, f"Arith not implemented for {a.type} and {b.type}")

@dp_inst
def _compile_arith(state, arith, a: T.any, b: T.any):
    raise Signal.make(T.TypeError, state, arith.op, f"Operator '{arith.op}' not implemented for {a.type} and {b.type}")

@dp_inst
def _compile_arith(state, arith, a: T.vectorized, b: T.vectorized):
    return vectorized(_compile_arith(state, arith, unvectorized(a), unvectorized(b)))

@dp_inst
def _compile_arith(state, arith, a: T.any, b: T.vectorized):
    return vectorized(_compile_arith(state, arith, a, unvectorized(b)))

@dp_inst
def _compile_arith(state, arith, a: T.vectorized, b: T.any):
    return vectorized(_compile_arith(state, arith, unvectorized(a), b))



@dp_inst
def _compile_arith(state, arith, a: T.collection, b: T.collection):
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
        raise Signal.make(T.TypeError, state, arith.op, f"Operation '{arith.op}' not supported for tables ({a.type}, {b.type})")

    return state.get_var(op).func(state, a, b)


@dp_inst
def _compile_arith(state, arith, a: T.aggregate, b: T.aggregate):
    res = _compile_arith(state, arith, a.elem, b.elem)
    return objects.aggregate(res)

@dp_inst
def _compile_arith(state, arith, a: T.aggregate, b: T.primitive):
    res = _compile_arith(state, arith, a.elem, b)
    return objects.aggregate(res)
@dp_inst
def _compile_arith(state, arith, a: T.primitive, b: T.aggregate):
    return _compile_arith(state, arith, b, a)

@dp_inst
def _compile_arith(state, arith, a: T.string, b: T.int):
    if arith.op != '*':
        raise Signal.make(T.TypeError, state, arith.op, f"Operator '{arith.op}' not supported between string and integer.")
    return call_pql_func(state, "repeat", [a, b])

@dp_inst
def _compile_arith(state, arith, a: T.number, b: T.number):
    if arith.op == '/' or a.type <= T.float or b.type <= T.float:
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
        }[arith.op]
    except KeyError:
        raise Signal.make(T.TypeError, state, arith, f"Operator {arith.op} not supported between types '{a.type}' and '{b.type}'")

    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        # Local folding for better performance.
        # However, acts a little different than SQL. For example, in this branch 1/0 raises ValueError,
        # while SQL returns NULL
        try:
            value = f(a.local_value, b.local_value)
        except ZeroDivisionError as e:
            raise Signal.make(T.ValueError, state, arith.args[-1], str(e))
        return new_value_instance(value, res_type)

    code = sql.arith(state.db.target, res_type, arith.op, [a.code, b.code])
    return make_instance(code, res_type, [a, b])

@dp_inst
def _compile_arith(state, arith, a: T.string, b: T.string):
    if arith.op == '~':
        code = sql.Like(a.code, b.code)
        return objects.Instance.make(code, T.bool, [a, b])

    if arith.op != '+':
        raise exc.Signal.make(T.TypeError, state, arith.op, f"Operator '{arith.op}' not supported for strings.")

    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        # Local folding for better performance (optional, for better performance)
        return new_value_instance(a.local_value + b.local_value, T.string)

    code = sql.arith(state.db.target, T.string, arith.op, [a.code, b.code])
    return make_instance(code, T.string, [a, b])



@dy
def compile_to_inst(state: State, x: ast.Ellipsis):
    raise Signal.make(T.SyntaxError, state, x, "Ellipsis not allowed here")


@dy
def compile_to_inst(state: State, c: ast.Const):
    if c.type == T.nulltype:
        assert c.value is None
        return objects.null
    return new_value_instance(c.value, c.type)

@dy
def compile_to_inst(state: State, d: ast.Dict_):
    # TODO handle duplicate key names
    elems = {k or guess_field_name(v): evaluate(state, v) for k, v in d.elems.items()}
    t = T.struct({k: v.type for k, v in elems.items()})
    return objects.StructInstance(t, elems)


@dy
def compile_to_inst(state: State, lst: objects.PythonList):
    t = lst.type.elem
    x = [sql.Primitive(t, sql._repr(t,i)) for i in (lst.items)]
    name = state.unique_name("list_")
    table_code, subq = sql.create_list(lst.type, name, x)
    inst = objects.ListInstance.make(table_code, lst.type, [])
    inst.subqueries[name] = subq
    return inst


@dy
def compile_to_inst(state: State, lst: ast.List_):
    # TODO generate (a,b,c) syntax for IN operations, with its own type
    # sql = "(" * join([e.code.text for e in objs], ",") * ")"
    # type = length(objs)>0 ? objs[1].type : nothing
    # return Instance(Sql(sql), ArrayType(type, false))
    # Or just evaluate?

    if not lst.elems and tuple(lst.type.elems.values()) == (T.any,):
        # XXX a little awkward
        return objects.EmptyList

    elems = evaluate(state, lst.elems)

    elem_type = union_types(e.type for e in elems)

    if not (elem_type <= T.union[T.primitive, T.nulltype]):
        raise Signal.make(T.TypeError, state, lst, "Cannot create lists of type %s" % elem_type)

    assert elem_type <= lst.type.elems[ITEM_NAME]

    list_type = T.list[elem_type]
    name = state.unique_name("list_")
    table_code, subq = sql.create_list(list_type, name, [e.code for e in elems])

    inst = objects.ListInstance.make(table_code, list_type, elems)
    inst.subqueries[name] = subq
    return inst


# def resolve_parameters(state: State, res: ast.ResolveParameters):
@dy
def compile_to_inst(state: State, res: ast.ResolveParameters):

    # XXX use a different mechanism??

    # basically cast_to_instance(). Ideally should be an instance whenever possible
    if isinstance(res.obj, objects.Instance):
        obj = res.obj
    else:
        with state.use_scope(res.values):
            obj = evaluate(state, res.obj)

        # handle non-compilable entities (meta, etc.)
        if not isinstance(obj, objects.Instance):
            if isinstance(obj, objects.Function):
                return obj
            return res.replace(obj=obj)

    state.require_access(state.AccessLevels.WRITE_DB)

    sq2 = SafeDict()
    code = _resolve_sql_parameters(state, obj.code, subqueries=sq2)
    subqueries = {k: _resolve_sql_parameters(state, v, subqueries=sq2) for k, v in obj.subqueries.items()}

    return obj.replace(code=code, subqueries=SafeDict(subqueries).update(sq2))


def _resolve_sql_parameters(state, compiled_sql, wrap=False, subqueries=None):
    qb = sql.QueryBuilder(state.db.target, False)

    # Ensure <= CompiledSQL
    compiled_sql = compiled_sql.compile(qb)

    new_code = []
    for c in compiled_sql.code:
        if isinstance(c, sql.Parameter):
            inst = evaluate(state, state.get_var(c.name))
            if inst.type != c.type:
                raise Signal.make(T.CastError, state, None, f"Internal error: Parameter is of wrong type ({c.type} != {inst.type})")
            new_code += inst.code.compile_wrap(qb).code
            subqueries.update(inst.subqueries)
        else:
            new_code.append(c)

    res = compiled_sql.replace(code=new_code)
    if wrap:
        res = res.wrap(qb)
    return res




@dy
def compile_to_inst(state: State, rps: ast.ParameterizedSqlCode):
    sql_code = cast_to_python(state, rps.string)
    if not isinstance(sql_code, str):
        raise Signal.make(T.TypeError, state, rps, f"Expected string, got '{rps.string}'")

    type_ = evaluate(state, rps.type)
    if isinstance(type_, objects.Instance):
        type_ = type_.type
    assert isinstance(type_, Type), type_
    name = state.unique_name("subq_")
    if type_ <= T.table:
        self_table = objects.new_table(type_, Id(name))
    else:
        self_table = None

    instances = []
    subqueries = SafeDict()
    tokens = re_split(r"\$\w+", sql_code)
    new_code = []
    for m, t in tokens:
        if m:
            assert t[0] == '$'
            if t == '$self':
                if self_table is None:
                    raise exc.Signal.make(T.TypeError, state, rps, f"$self is only available for queries that return a table")
                inst = self_table
            else:
                obj = state.get_var(t[1:])
                if isinstance(obj, Type) and obj <= T.table:
                    # This branch isn't strictly necessary
                    # It exists to create nicer SQL code output
                    inst = objects.new_table(obj)
                else:
                    inst = cast_to_instance(state, obj)

            instances.append(inst)
            new_code += _resolve_sql_parameters(state, inst.code, wrap=bool(new_code), subqueries=subqueries).code
            assert not subqueries
        else:
            new_code.append(t)

    # TODO validation!!
    if type_ <= T.table:
        code = sql.CompiledSQL(type_, new_code, None, True, False)

        # TODO this isn't in the tests!
        fields = [sql.Name(c, path) for path, c in flatten_type(type_)]

        subq = sql.Subquery(name, fields, code)

        inst = objects.new_table(type_, Id(name), instances)
        inst.subqueries[name] = subq
        return inst

    code = sql.CompiledSQL(type_, new_code, None, False, False)     # XXX is False correct?
    return make_instance(code, type_, instances)

@dy
def compile_to_inst(state: State, s: ast.Slice):
    obj = cast_to_instance(state, s.obj)

    assert_type(obj.type, T.union[T.string, T.collection, T.vectorized[T.string]], state, s, "Slice")

    instances = [obj]
    if s.range.start:
        start = cast_to_instance(state, s.range.start)
        instances += [start]
    else:
        start = new_value_instance(0)

    if s.range.stop:
        stop = cast_to_instance(state, s.range.stop)
        instances += [stop]
    else:
        stop = None

    if obj.type <= T.string or obj.type <= T.vectorized[T.string]:
        code = sql.StringSlice(obj.code, sql.add_one(start.code), stop and sql.add_one(stop.code))
    else:
        start_n = cast_to_python(state, start)
        stop_n = stop and cast_to_python(state, stop)
        code = sql.table_slice(obj, start_n, stop_n)

    return make_instance(code, obj.type, instances)

@dy
def compile_to_inst(state: State, sel: ast.Selection):
    obj = simplify(state, sel.table)
    if isinstance(obj, Type):
        return _apply_type_generics(state, obj, sel.conds)

    table = cast_to_instance(state, obj)

    if table.type <= T.string or table.type <= T.vectorized[T.string]:
        # raise exc.Signal.make(T.NotImplementedError, state, sel, "String indexing not implemented yet. Use slicing instead (s[start..stop])")
        index ,= sel.conds
        assert index.type <= T.int
        table = table.replace(type=T.string)    # XXX why get rid of vectorized here? because it's a table operation node?
        slice = ast.Slice(table, ast.Range(index, ast.Arith('+', [index, ast.Const(T.int, 1)]))).set_text_ref(sel.text_ref)
        return compile_to_inst(state, slice)

    assert_type(table.type, T.collection, state, sel, "Selection")

    with state.use_scope({n:vectorized(c) for n, c in table.all_attrs().items()}):
        conds = cast_to_instance(state, sel.conds)

    if any(t <= T.unknown for t in table.type.elem_types):
        code = sql.unknown
    else:
        for i, c in enumerate(conds):
            if not (c.type <= T.union[T.bool, T.vectorized[T.bool]]):
                raise exc.Signal.make(T.TypeError, state, sel.conds[i], f"Selection expected boolean, got {c.type}")

        code = sql.table_selection(table, [c.code for c in conds])

    return type(table).make(code, table.type, [table] + conds)

@dy
def compile_to_inst(state: State, param: ast.Parameter):
    if state.access_level == state.AccessLevels.COMPILE:
        if param.type <= T.struct:
            # TODO why can't I just make an instance?
            raise exc.InsufficientAccessLevel("Structs not supported yet")
        return make_instance(sql.Parameter(param.type, param.name), param.type, [])
    else:
        return state.get_var(param.name)

@dy
def compile_to_inst(state: State, attr: ast.Attr):
    if isinstance(attr.name, ast.Marker):
        if attr.expr:
            inst = evaluate(state, attr.expr)
            attrs = {k:(0,v) for k, v in inst.all_attrs().items()}
        else:
            attrs = {}
        raise AutocompleteSuggestions(attrs)

    if not attr.expr:
        raise Signal.make(T.NotImplementedError, state, attr, "Implicit attribute syntax not supported")

    inst = evaluate(state, attr.expr)
    try:
        return evaluate(state, inst.get_attr(attr.name))
    except exc.pql_AttributeError as e:
        raise Signal.make(T.AttributeError, state, attr, e.message)



def _apply_type_generics(state, gen_type, type_names):
    type_objs = evaluate(state, type_names)
    if not type_objs:
        raise Signal.make(T.TypeError, state, None, f"Generics expression expected a type, got nothing.")
    for o in type_objs:
        if not isinstance(o, Type):
            raise Signal.make(T.TypeError, state, None, f"Generics expression expected a type, got '{o}'.")

    if len(type_objs) > 1:
        if gen_type in (T.union,):
            return gen_type(tuple(type_objs))

        raise Signal.make(T.TypeError, state, None, "Union types not yet supported!")
    else:
        t ,= type_objs

    try:
        return gen_type[t]
    except TypeError:
        raise Signal.make(T.TypeError, state, None, f"Type {t} isn't a container!")



@dy
def guess_field_name(f):
    return '_'
@dy
def guess_field_name(f: ast.Attr):
    return guess_field_name(f.expr) + "." + f.name
@dy
def guess_field_name(f: ast.Name):
    return str(f.name)
@dy
def guess_field_name(f: ast.Projection):
    return guess_field_name(f.table)
@dy
def guess_field_name(f: ast.FuncCall):
    return guess_field_name(f.func)


@dy
def compile_to_inst(state: State, marker: ast.Marker):
    all_vars = state.get_all_vars_with_rank()   # Uses overridden version of AcState
    raise AutocompleteSuggestions(all_vars)

@dy
def compile_to_inst(state: State, range: ast.Range):
    start = cast_to_python(state, range.start) if range.start else 0
    if not isinstance(start, int):
        raise Signal.make(T.TypeError, state, range, "Range must be between integers")
    if range.stop:
        stop = cast_to_python(state, range.stop)
        if not isinstance(stop, int):
            raise Signal.make(T.TypeError, state, range, "Range must be between integers")
        stop_str = f" WHERE item+1<{stop}"
    else:
        if state.db.target is sql.mysql:
            raise Signal.make(T.NotImplementedError, state, range, "MySQL doesn't support infinite recursion!")
        stop_str = ''

    type_ = T.list[T.int]
    name = state.unique_name("range")
    skip = 1
    code = f"SELECT {start} AS item UNION ALL SELECT item+{skip} FROM {name}{stop_str}"
    subq = sql.Subquery(name, [], sql.RawSql(type_, code))
    code = sql.TableName(type_, Id(name))
    return objects.ListInstance(code, type_, SafeDict({name: subq}))
