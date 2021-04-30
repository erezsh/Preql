import operator


from preql import settings
from preql.utils import safezip, listgen, find_duplicate, SafeDict, re_split
from preql.context import context

from .exceptions import Signal, InsufficientAccessLevel, ReturnSignal, pql_AttributeError
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import dsp, State, assert_type, pyvalue_inst, evaluate, simplify, call_builtin_func, cast_to_python_string, cast_to_python_int
from .pql_types import T, Type, Id, ITEM_NAME, dp_inst
from .types_impl import flatten_type, pql_repr, kernel_type
from .casts import cast
from .pql_objects import AbsInstance, projected, make_instance, remove_phantom_type

class AutocompleteSuggestions(Exception):
    pass

@dsp
def cast_to_instance(x: list):
    return [cast_to_instance(i) for i in x]

@dsp
def cast_to_instance(x):
    try:
        x = simplify(x)  # just compile Name?
        inst = compile_to_inst(x)
        # inst = evaluate( x)
    except ReturnSignal:
        raise Signal.make(T.CompileError, None, f"Bad compilation of {x}")

    if isinstance(inst, ast.ParameterizedSqlCode):
        raise InsufficientAccessLevel(inst)

    if not isinstance(inst, AbsInstance):
        # TODO compile error? cast error?
        # TODO need to be able to catch this above, and provide better errors
        raise Signal.make(T.TypeError, None, f"Could not compile {pql_repr(inst.type, inst)}")

    return inst



@listgen
def _process_fields(fields):
    for f in fields:
        try:
            v = cast_to_instance(f.value)
        except Signal as e:
            if e.type <= T.TypeError:
                raise e.replace(message=f"Cannot use object of type '{evaluate( f.value).type}' in projection.")
            raise

        # In Preql, {=>v} creates an array. In SQL, it selects the first element.
        # Here we mitigate that disparity.
        if v.type <= T.aggregated:
            v = v.primary_key()
            t = T.json_array[v.type]
            v = make_instance(sql.MakeArray(t, v.code), t, [v])

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name

        yield [(f.user_defined and bool(f.name), name), v]


def _exclude_items(d, req_exclude, opt_exclude):
    if not req_exclude <= set(d):
        raise ValueError(req_exclude - set(d))

    for k, v in d.items():
        if k not in req_exclude and k not in opt_exclude:
            yield k, v


@listgen
def _expand_ellipsis(obj, fields):
    direct_names = {f.value.name for f in fields if isinstance(f.value, ast.Name)}

    for f in fields:
        assert isinstance(f, ast.NamedField)

        if isinstance(f.value, ast.Ellipsis):
            if f.name:
                msg = "Cannot use a name for ellipsis (inlining operation doesn't accept a name)"
                raise Signal.make(T.SyntaxError, f, msg)

            t = obj.type
            assert t <= T.table or t <= T.struct # some_table{ ... } or some_table{ some_struct_item {...} }

            for n in f.value.exclude:
                if isinstance(n, ast.Marker):
                    raise AutocompleteSuggestions({k:(0, v) for k, v in t.elems.items()
                                                   if k not in direct_names
                                                   and k not in f.value.exclude})

                if n in direct_names:
                    raise Signal.make(T.NameError, n, f"Field to exclude '{n}' is explicitely included in projection")

            if f.value.from_struct:
                # Inline struct
                with context.state.use_scope(obj.all_attrs()):
                    s = evaluate( f.value.from_struct)
                    items = s.attrs
            else:
                # Ellipsis for current projection
                items = obj.all_attrs()

            try:
                remaining_items = list(_exclude_items(items, set(f.value.exclude), direct_names))
            except ValueError as e:
                fte = set(e.args[0])
                raise Signal.make(T.NameError, obj, f"Fields to exclude '{fte}' not found")

            exclude = direct_names | set(f.value.exclude)
            for name, value in remaining_items:
                assert isinstance(name, str)
                assert name not in exclude
                yield ast.NamedField(name, value, user_defined=False).set_text_ref(f.text_ref)
        else:
            yield f


@dsp
def compile_to_inst(x):
    return x
@dsp
def compile_to_inst(node: ast.Ast):
    return node


@dsp
def compile_to_inst(cb: ast.CodeBlock):
    if len(cb.statements) == 1:
        return compile_to_inst(cb.statements[0])

    # TODO some statements can be evaluated at compile time
    raise Signal.make(T.CompileError, cb, "Cannot compile this code block")
@dsp
def compile_to_inst(i: ast.If):
    cond = cast(cast_to_instance(i.cond), T.bool)
    then = cast_to_instance(i.then)
    else_ = cast_to_instance(i.else_)
    code = sql.Case(cond.code, then.code, else_.code)
    # TODO simplify this with a better type system
    res_type = kernel_type(then.type) | kernel_type(else_.type)
    inst = make_instance(code, res_type, [cond, then, else_])
    return objects.inherit_phantom_type(inst, [cond, then, else_])



@dsp
def compile_to_inst(proj: ast.Projection):
    state = context.state
    table = cast_to_instance(proj.table)

    if table is objects.EmptyList:
        return table   # Empty list projection is always an empty list.

    t = T.union[T.table, T.struct]
    if not table.type <= t:
        raise Signal.make(T.TypeError, proj, f"Cannot project objects of type {table.type}")

    fields = _expand_ellipsis(table, proj.fields)

    # Test duplicates in field names. If an automatic name is used, collision should be impossible
    dup = find_duplicate([f for f in list(proj.fields) + list(proj.agg_fields) if f.name], key=lambda f: f.name)
    if dup:
        raise Signal.make(T.TypeError, dup, f"Field '{dup.name}' was already used in this projection")

    attrs = table.all_attrs()

    with state.use_scope({n: projected(c) for n, c in attrs.items()}):
        fields = _process_fields(fields)

    for name, f in fields:
        if not f.type <= T.union[T.primitive, T.struct, T.json, T.nulltype, T.unknown]:
            raise Signal.make(T.TypeError, proj, f"Cannot project values of type: {f.type}")

    if isinstance(table, objects.StructInstance):
        d = {n[1]:c for n, c in fields}     # Remove used_defined bool
        t = T.struct({n:f.type for n, f in d.items()})
        return objects.StructInstance(t, d)

    agg_fields = []
    if proj.agg_fields:
        with state.use_scope({n:objects.aggregate(c) for n, c in attrs.items()}):
            agg_fields = _process_fields(proj.agg_fields)

    all_fields = fields + agg_fields
    assert all(isinstance(inst, AbsInstance) for name_, inst in all_fields)

    #
    # Make new type (and resolve names)
    #
    field_types = [(name, inst.type) for name, inst in all_fields]
    reserved_names = {name[1] for name, _ in all_fields if name[0]}
    elems = {}
    for (user_defined, name), type_ in field_types:
        # Unvectorize for placing in the table type
        type_ = kernel_type(type_)

        # Find name without collision
        if not user_defined:
            name_ = name
            i = 1
            while name in elems or name in reserved_names:
                name = name_ + str(i)
                i += 1

        assert name not in elems
        elems[name] = type_

    # TODO inherit primary key? indexes?
    # codename = state.unique_name('proj')
    new_table_type = T.table(elems, temporary=False)    # XXX abstract=True

    # Make code
    flat_codes = [code
                  for _, inst in all_fields
                  for code in inst.flatten_code()]

    sql_fields = [
        sql.ColumnAlias.make(code, nn)
        for code, (nn, _nt) in safezip(flat_codes, flatten_type(new_table_type))
    ]

    if not sql_fields:
        raise Signal.make(T.TypeError, proj, "No column provided for projection (empty projection)")

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

@dsp
def compile_to_inst(order: ast.Order):
    table = cast_to_instance(order.table)
    assert_type(table.type, T.table, order, "'order'")

    with context.state.use_scope(table.all_attrs()):
        fields = cast_to_instance(order.fields)

    for f in fields:
        if not f.type <= T.primitive:
            # TODO Support 'ordering' trait?
            raise Signal.make(T.TypeError, order, f"Arguments to 'order' must be primitive")


    code = sql.table_order(table, [c.code for c in fields])

    return objects.TableInstance.make(code, table.type, [table] + fields)

@dsp
def compile_to_inst(expr: ast.DescOrder):
    obj = cast_to_instance(expr.value)
    return obj.replace(code=sql.Desc(obj.code))



@dsp
def compile_to_inst(lst: list):
    return [evaluate( e) for e in lst]


@dsp
def compile_to_inst(o: ast.Or):
    args = cast_to_instance(o.args)
    a, b = args
    if kernel_type(a.type) != kernel_type(b.type):
        msg = f"'or' operator requires both arguments to be of the same type, got {a.type} and {b.type}"
        raise Signal.make(T.TypeError, o, msg)
    cond = cast(a, T.bool)
    code = sql.Case(cond.code, a.code, b.code)
    return objects.make_instance(code, a.type, args)

@dsp
def compile_to_inst(o: ast.And):
    args = cast_to_instance(o.args)
    a, b = args
    if kernel_type(a.type) != kernel_type(b.type):
        msg = f"'and' operator requires both arguments to be of the same type, got {a.type} and {b.type}"
        raise Signal.make(T.TypeError, o, msg)
    cond = cast(a, T.bool)
    code = sql.Case(cond.code, b.code, a.code)
    return objects.make_instance(code, a.type, args)

@dsp
def compile_to_inst(o: ast.Not):
    expr = cast_to_instance(o.expr)
    expr_bool = cast(expr, T.bool)
    code = sql.LogicalNot(expr_bool.code)
    return objects.make_instance(code, T.bool, [expr])



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
def _prepare_to_compare(op, a: T.number | T.bool | T.t_id, b: T.number | T.bool | T.t_id):
    pass
@dp_inst
def _prepare_to_compare(op, a: T.string, b: T.string):
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



@dsp
def compile_to_inst(cmp: ast.Compare):
    insts = evaluate( cmp.args)

    if cmp.op == 'in' or cmp.op == '!in':
        return contains(cmp.op, insts[0], insts[1])

    op = {
        '==': '=',
        '<>': '!=',
    }.get(cmp.op, cmp.op)
    return compare(op, insts[0], insts[1])

@dsp
def compile_to_inst(neg: ast.Neg):
    expr = cast_to_instance(neg.expr)
    assert_type(expr.type, T.number, neg, "Negation")

    return make_instance(sql.Neg(expr.code), expr.type, [expr])


@dsp
def compile_to_inst(arith: ast.BinOp):
    args = cast_to_instance(arith.args)
    return compile_arith(arith, *args)


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

    return context.state.get_var(op).func(a, b)



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

    code = sql.arith(context.state.db.target, res_type, arith.op, [a.code, b.code])
    return make_instance(code, res_type, [a, b])

@dp_inst
def _compile_arith(arith, a: T.string, b: T.string):
    if arith.op == 'like':
        code = sql.BinOp('like', [a.code, b.code])
        return objects.Instance.make(code, T.bool, [a, b])

    if arith.op != '+':
        raise Signal.make(T.TypeError, arith.op, f"Operator '{arith.op}' not supported for strings.")

    if settings.optimize and isinstance(a, objects.ValueInstance) and isinstance(b, objects.ValueInstance):
        # Local folding for better performance (optional, for better performance)
        return pyvalue_inst(a.local_value + b.local_value, T.string)

    code = sql.arith(context.state.db.target, T.string, arith.op, [a.code, b.code])
    return make_instance(code, T.string, [a, b])



@dsp
def compile_to_inst(x: ast.Ellipsis):
    raise Signal.make(T.SyntaxError, x, "Ellipsis not allowed here")


@dsp
def compile_to_inst(c: ast.Const):
    if c.type == T.nulltype:
        assert c.value is None
        return objects.null
    return pyvalue_inst(c.value, c.type)

@dsp
def compile_to_inst(d: ast.Dict_):
    # TODO handle duplicate key names
    elems = {k or guess_field_name(v): evaluate( v) for k, v in d.elems.items()}
    t = T.struct({k: v.type for k, v in elems.items()})
    return objects.StructInstance(t, elems)


@dsp
def compile_to_inst(lst: objects.PythonList):
    t = lst.type.elem
    x = [sql.Primitive(t, sql._repr(t,i)) for i in lst.items]
    name = context.state.unique_name("list_")
    table_code, subq, list_type = sql.create_list(name, x)
    inst = objects.TableInstance.make(table_code, list_type, [])
    inst.subqueries[name] = subq
    return inst


@dsp
def compile_to_inst(lst: ast.List_):
    if not lst.elems and tuple(lst.type.elems.values()) == (T.any,):
        # XXX a little awkward
        return objects.EmptyList

    elems = evaluate( lst.elems)

    types = {e.type for e in elems}

    if len(types) > 1:
        raise Signal.make(T.TypeError, lst, f"List members must be of the same type. Got {types}")

    elem_type ,= types

    if elem_type <= T.struct:
        rows = [sql.ValuesTuple(obj.type, obj.flatten_code()) for obj in elems]
        list_type = T.table(elems=elem_type.elems)
        name = context.state.unique_name("table_")
        table_code, subq = sql.create_table(list_type, name, rows)
    else:
        if not (elem_type <= T.union[T.primitive, T.nulltype]):
            raise Signal.make(T.TypeError, lst, "Cannot create lists of type %s" % elem_type)

        assert elem_type <= lst.type.elems[ITEM_NAME], (elem_type, lst.type)

        name = context.state.unique_name("list_")
        table_code, subq, list_type = sql.create_list(name, [e.code for e in elems])

    inst = objects.TableInstance.make(table_code, list_type, elems)
    inst.subqueries[name] = subq
    return inst


# def resolve_parameters(state: State, res: ast.ResolveParameters):
@dsp
def compile_to_inst(res: ast.ResolveParameters):
    state = context.state

    # XXX use a different mechanism??

    # basically cast_to_instance(). Ideally should be an instance whenever possible
    if isinstance(res.obj, objects.Instance):
        obj = res.obj
    else:
        with state.use_scope(res.values):
            obj = evaluate( res.obj)

        # handle non-compilable entities (meta, etc.)
        if not isinstance(obj, objects.Instance):
            if isinstance(obj, objects.Function):
                return obj
            return res.replace(obj=obj)

    state.require_access(state.AccessLevels.WRITE_DB)

    sq2 = SafeDict()
    code = _resolve_sql_parameters(obj.code, subqueries=sq2)
    subqueries = {k: _resolve_sql_parameters(v, subqueries=sq2) for k, v in obj.subqueries.items()}

    return obj.replace(code=code, subqueries=SafeDict(subqueries).update(sq2))


def _resolve_sql_parameters(compiled_sql, wrap=False, subqueries=None):
    state = context.state
    qb = sql.QueryBuilder(state.db.target, False)

    # Ensure <= CompiledSQL
    compiled_sql = compiled_sql.compile(qb)

    new_code = []
    for c in compiled_sql.code:
        if isinstance(c, sql.Parameter):
            inst = evaluate( state.get_var(c.name))
            if inst.type != c.type:
                msg = f"Internal error: Parameter is of wrong type ({c.type} != {inst.type})"
                raise Signal.make(T.CastError, None, msg)
            new_code += inst.code.compile_wrap(qb).code
            subqueries.update(inst.subqueries)
        else:
            new_code.append(c)

    res = compiled_sql.replace(code=new_code)
    if wrap:
        res = res.wrap(qb)
    return res




@dsp
def compile_to_inst(rps: ast.ParameterizedSqlCode):
    state = context.state 

    sql_code = cast_to_python_string(rps.string)
    if not isinstance(sql_code, str):
        raise Signal.make(T.TypeError, rps, f"Expected string, got '{rps.string}'")

    type_ = evaluate( rps.type)
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
                    raise Signal.make(T.TypeError, rps, f"$self is only available for queries that return a table")
                inst = self_table
            else:
                obj = state.get_var(t[1:])
                if isinstance(obj, Type) and obj <= T.table:
                    # This branch isn't strictly necessary
                    # It exists to create nicer SQL code output
                    inst = objects.new_table(obj)
                else:
                    inst = cast_to_instance(obj)

            instances.append(inst)
            new_code += _resolve_sql_parameters(inst.code, wrap=bool(new_code), subqueries=subqueries).code
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

    # Cannot inherit aggregated, because some expressions such as `sum()` cancel it out.
    # XXX So why is it okay for projected?
    #     Maybe all results should be projected?
    type_ = objects.inherit_vectorized_type(type_, instances)
    code = sql.CompiledSQL(type_, new_code, None, False, False)     # XXX is False correct?
    return make_instance(code, type_, instances)

@dsp
def compile_to_inst(s: ast.Slice):
    obj = cast_to_instance(s.obj)

    assert_type(obj.type, T.union[T.string, T.table], s, "Slice")

    instances = [obj]
    if s.range.start:
        start = cast_to_instance(s.range.start)
        instances += [start]
    else:
        start = pyvalue_inst(0)

    if s.range.stop:
        stop = cast_to_instance(s.range.stop)
        instances += [stop]
    else:
        stop = None

    if obj.type <= T.string:
        code = sql.StringSlice(obj.code, sql.add_one(start.code), stop and sql.add_one(stop.code))
    else:
        start_n = cast_to_python_int(start)
        stop_n = stop and cast_to_python_int(stop)
        code = sql.table_slice(obj, start_n, stop_n)

    return make_instance(code, obj.type, instances)

@dsp
def compile_to_inst(sel: ast.Selection):
    obj = simplify(sel.table)
    if isinstance(obj, Type):
        return _apply_type_generics(obj, sel.conds)

    table = cast_to_instance(obj)

    if table.type <= T.string:
        index ,= cast_to_instance(sel.conds)
        assert index.type <= T.int, index.type
        table = table.replace(type=T.string)    # XXX why get rid of projected here? because it's a table operation node?
        slice = ast.Slice(table,
                          ast.Range(index, ast.BinOp('+', [index, ast.Const(T.int, 1)]))
                         ).set_text_ref(sel.text_ref)
        return compile_to_inst(slice)

    assert_type(table.type, T.table, sel, "Selection")

    with context.state.use_scope({n:projected(c) for n, c in table.all_attrs().items()}):
        conds = cast_to_instance(sel.conds)

    if any(t <= T.unknown for t in table.type.elem_types):
        code = sql.unknown
    else:
        for i, c in enumerate(conds):
            if not c.type <= T.bool:
                raise Signal.make(T.TypeError, sel.conds[i], f"Selection expected boolean, got {c.type}")

        code = sql.table_selection(table, [c.code for c in conds])

    return type(table).make(code, table.type, [table] + conds)

@dsp
def compile_to_inst(param: ast.Parameter):
    state = context.state
    if state.access_level == state.AccessLevels.COMPILE:
        if param.type <= T.struct:
            # TODO why can't I just make an instance?
            raise InsufficientAccessLevel("Structs not supported yet")
        return make_instance(sql.Parameter(param.type, param.name), param.type, [])

    return state.get_var(param.name)

@dsp
def compile_to_inst(attr: ast.Attr):
    if isinstance(attr.name, ast.Marker):
        if attr.expr:
            inst = evaluate( attr.expr)
            attrs = {k:(0,v) for k, v in inst.all_attrs().items()}
        else:
            attrs = {}
        raise AutocompleteSuggestions(attrs)

    if not attr.expr:
        raise Signal.make(T.NotImplementedError, attr, "Implicit attribute syntax not supported")

    inst = evaluate( attr.expr)
    try:
        return evaluate( inst.get_attr(attr.name))
    except pql_AttributeError as e:
        raise Signal.make(T.AttributeError, attr, e.message)



def _apply_type_generics(gen_type, type_names):
    type_objs = evaluate( type_names)
    if not type_objs:
        raise Signal.make(T.TypeError, None, f"Generics expression expected a type, got nothing.")
    for o in type_objs:
        if not isinstance(o, Type):
            if isinstance(o.code, sql.Parameter):
                # XXX hacky test, hacky solution
                raise InsufficientAccessLevel()
            raise Signal.make(T.TypeError, None, f"Generics expression expected a type, got '{o}'.")

    if len(type_objs) > 1:
        if gen_type in (T.union,):
            return gen_type(tuple(type_objs))

        raise Signal.make(T.TypeError, None, "Union types not yet supported!")

    t ,= type_objs
    try:
        return gen_type[t]
    except TypeError:
        raise Signal.make(T.TypeError, None, f"Type {t} isn't a container!")



@dsp
def guess_field_name(_f):
    return '_'
@dsp
def guess_field_name(f: ast.Attr):
    name = f.name
    if isinstance(name, ast.Marker):
        name = '<marker>'
    return guess_field_name(f.expr) + "." + name
@dsp
def guess_field_name(f: ast.Name):
    return str(f.name)
@dsp
def guess_field_name(f: ast.Projection):
    return guess_field_name(f.table)
@dsp
def guess_field_name(f: ast.FuncCall):
    return guess_field_name(f.func)


@dsp
def compile_to_inst(marker: ast.Marker):
    all_vars = context.state.get_all_vars_with_rank()   # Uses overridden version of AcState
    raise AutocompleteSuggestions(all_vars)

@dsp
def compile_to_inst(range: ast.Range):
    state = context.state
    # TODO move to sql.py
    # Requires subqueries to be part of 'code' instead of a separate 'subqueries'?
    # But then what's the point of an instance, other than carrying methods...
    start = cast_to_python_int(range.start) if range.start else 0
    if not isinstance(start, int):
        raise Signal.make(T.TypeError, range, "Range must be between integers")

    stop = None
    if range.stop:
        stop = cast_to_python_int(range.stop)
        if not isinstance(stop, int):
            raise Signal.make(T.TypeError, range, "Range must be between integers")
    elif state.db.target in (sql.mysql, sql.bigquery):
            raise Signal.make(T.NotImplementedError, range, f"{state.db.target} doesn't support infinite series!")

    type_ = T.list[T.int]

    if state.db.target == sql.bigquery:
        code = sql.RawSql(type_, f'UNNEST(GENERATE_ARRAY({start}, {stop-1})) as item')
        return objects.TableInstance.make(code, type_, [])

    if stop is None:
        stop_str = ''
    else:
        stop_str = f" WHERE item+1<{stop}"


    name = state.unique_name("range")
    skip = 1
    code = f"SELECT {start} AS item UNION ALL SELECT item+{skip} FROM {name}{stop_str}"
    subq = sql.Subquery(name, [], sql.RawSql(type_, code))
    code = sql.TableName(type_, Id(name))
    return objects.TableInstance(code, type_, SafeDict({name: subq}))


