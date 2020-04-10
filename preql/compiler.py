from .utils import safezip, listgen, SafeDict, find_duplicate
from .exceptions import pql_TypeError, PreqlError, pql_AttributeError, pql_SyntaxError, pql_CompileError
from . import exceptions as exc

from . import settings
from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import dy, State, assert_type, new_value_instance, evaluate, call_pql_func


@dy
def compile_type_def(state: State, table: types.TableType) -> sql.Sql:
    posts = []
    pks = []
    columns = []

    pks = {types.join_names(pk) for pk in table.primary_keys}
    for name, c in table.flatten_type():
        type_ = compile_type(state, c)
        columns.append( f"{name} {type_}" )
        if isinstance(c, types.RelationalColumn):
            # TODO any column, using projection / get_attr
            if not table.temporary:
                # In postgres, constraints on temporary tables may reference only temporary tables
                s = f"FOREIGN KEY({name}) REFERENCES {c.type.name}(id)"
                posts.append(s)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    command = "CREATE TEMPORARY TABLE" if table.temporary else "CREATE TABLE IF NOT EXISTS"
    return sql.RawSql(types.null, f"{command} {table.name} (" + ", ".join(columns + posts) + ")")

@dy
def compile_type(state: State, type_: types.RelationalColumn):
    # TODO might have a different type
    return 'INTEGER'    # Foreign-key is integer

@dy
def compile_type(state: State, type_: types.DatumColumn):
    return compile_type(state, type_.type)

def _compile_type_primitive(type, nullable):
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'text': "TEXT",
        'datetime': "TIMESTAMP",
    }[type.name]
    if not nullable:
        s += " NOT NULL"
    return s
@dy
def compile_type(state: State, type: types.Primitive, nullable=False):
    return _compile_type_primitive(type, nullable)

@dy
def compile_type(state: State, type: types.OptionalType):
    return compile_type(state, type.type, True)

@dy
def compile_type(state: State, idtype: types.IdType, nullable=False):
    if state.db.target == sql.postgres:
        s = "SERIAL" # Postgres
    else:
        s = "INTEGER"
    if not nullable:
        s += " NOT NULL"
    return s


def _process_fields(state: State, fields):
    processed_fields = []
    for f in fields:

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name

        v = evaluate(state, f.value)

        if isinstance(v, ast.ResolveParametersString):
            raise exc.InsufficientAccessLevel()

        if not isinstance(v, objects.AbsInstance):
            raise pql_TypeError(None, f"Projection field is not an instance. Instead it is: {v}")

        if isinstance(v.type, types.Aggregated):
            v = v.primary_key()
            v = objects.make_instance(sql.MakeArray(v.type, v.code), v.type, [v])

        processed_fields.append( [name, v] )

    return processed_fields


@listgen
def _expand_ellipsis(table, fields):
    direct_names = {f.value.name for f in fields if isinstance(f.value, ast.Name)}

    for f in fields:
        assert isinstance(f, ast.NamedField)

        if isinstance(f.value, ast.Ellipsis):
            if f.name:
                raise pql_SyntaxError(f.meta, "Cannot use a name for ellipsis (inlining operation doesn't accept a name)")
            else:
                exclude = direct_names | set(f.value.exclude)
                for name in table.type.columns:
                    if name not in exclude:
                        yield ast.NamedField(f.meta, name, ast.Name(None, name))
        else:
            yield f


@dy
def compile_remote(state: State, x):
    return x
@dy
def compile_remote(state: State, node: ast.Ast):
    return node

@dy
def compile_remote(state: State, proj: ast.Projection):
    table = evaluate(state, proj.table)
    # assert_type(proj.meta, table.type, (types.TableType, types.ListType, types.StructType), "%s")
    if table is objects.EmptyList:
        return table   # Empty list projection is always an empty list.

    assert isinstance(table, objects.AbsInstance), table

    if not isinstance(table.type, (types.Collection, types.StructType)):
        raise pql_TypeError(proj.meta, f"Cannot project objects of type {table.type}")

    fields = _expand_ellipsis(table, proj.fields)

    # Test duplicates in field names. If an automatic name is used, collision should be impossible
    dup = find_duplicate([f for f in list(proj.fields) + list(proj.agg_fields) if f.name], key=lambda f: f.name)
    if dup:
        raise pql_TypeError(dup.meta, f"Field '{dup.name}' was already used in this projection")

    attrs = table.all_attrs()

    with state.use_scope(attrs):
        fields = _process_fields(state, fields)

    if isinstance(table, objects.StructInstance):
        t = types.StructType(state.unique_name('_nested_proj'), {n:c.type for n,c in fields})
        return objects.StructInstance(t, dict(fields))

    agg_fields = []
    if proj.agg_fields:
        with state.use_scope({n:objects.aggregated(c) for n,c in attrs.items()}):
            agg_fields = _process_fields(state, proj.agg_fields)

    all_fields = fields + agg_fields

    # Make new type
    new_table_type = types.TableType(state.unique_name(table.type.name + "_proj"), SafeDict(), True, [], code_prefix=state.unique_name('proj_')) # Maybe wrong
    for name_, inst in all_fields:
        assert isinstance(inst, objects.AbsInstance)

        # TODO what happens if automatic name preceeds and collides with user-given name?
        name = name_
        i = 1
        while name in new_table_type.columns:
            name = name_ + str(i)
            i += 1
        new_table_type.columns[name] = inst.type

    # Make code
    flat_codes = [code
                    for _, inst in all_fields
                    for code in inst.flatten_code()]

    # TODO if nn != on
    sql_fields = [
        sql.ColumnAlias.make(code, nn)
        for code, (nn, _nt) in safezip(flat_codes, new_table_type.flatten_type())
    ]

    # Make Instance
    new_table = objects.TableInstance.make(sql.null, new_table_type, [table])

    groupby = []
    if proj.groupby and fields:
        groupby = [new_table.get_column(n).primary_key().code for n, rc in fields]

    code = sql.Select(new_table_type, table.code, sql_fields, group_by=groupby)

    # Make Instance
    return new_table.replace(code=code)

@dy
def compile_remote(state: State, order: ast.Order):
    table = evaluate(state, order.table)
    assert_type(order.meta, table.type, types.TableType, "'order' expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.all_attrs()):
        fields = evaluate(state, order.fields)

    code = sql.table_order(table, [c.code for c in fields])

    return objects.TableInstance.make(code, table.type, [table] + fields)

@dy
def compile_remote(state: State, expr: ast.DescOrder):
    obj = evaluate(state, expr.value)
    return obj.replace(code=sql.Desc(obj.code))



@dy
def compile_remote(state: State, lst: list):
    return [evaluate(state, e) for e in lst]


@dy
def compile_remote(state: State, like: ast.Like):
    s = evaluate(state, like.str)
    p = evaluate(state, like.pattern)
    if s.type != types.String:
        raise pql_TypeError(like.str.meta.replace(parent=like.meta), f"Like (~) operator expects two strings")
    if p.type != types.String:
        raise pql_TypeError(like.pattern.meta.replace(parent=like.meta), f"Like (~) operator expects two strings")

    code = sql.Like(s.code, p.code)
    return objects.Instance.make(code, types.Bool, [s, p])


@dy
def compile_remote(state: State, cmp: ast.Compare):
    insts = evaluate(state, cmp.args)
    assert all(isinstance(i, objects.AbsInstance) for i in insts), (cmp.args, state.access_level)

    if cmp.op == 'in' or cmp.op == '!in':
        sql_cls = sql.Contains
        if isinstance(insts[0].type, types._String) and isinstance(insts[1].type, types._String):
            code = sql.Compare('>', [sql.FuncCall(types.Bool, 'INSTR', [i.code for i in reversed(insts)]), sql.value(0)])
            return objects.make_instance(code, types.Bool, [])
        else:
            assert_type(cmp.meta, insts[0].type, types.AtomicType, "Expecting type %s, got %s")
            assert_type(cmp.meta, insts[1].type, types.Collection, "Expecting type %s, got %s")
            cols = insts[1].type.columns
            if len(cols) > 1:
                raise pql_TypeError(cmp.meta, "Contains operator expects a collection with only 1 column! (Got %d)" % len(cols))
            c_type = list(cols.values())[0]
            if c_type.effective_type() != insts[0].type.effective_type():
                raise pql_TypeError(cmp.meta, f"Contains operator expects all types to match: {c_type} -- {insts[0].type}")

    else:

        sql_cls = sql.Compare
        # TODO should be able to coalesce, int->float, id->int, etc.
        #      also different types should still be comparable to some degree?
        # type_set = {i.type for i in insts}
        # if len(type_set) > 1:
        #     raise pql_TypeError(cmp.meta, "Cannot compare two different types: %s" % type_set)
        for i, inst in enumerate(insts):
            if not isinstance(inst.type.actual_type(), types.AtomicType):
                raise pql_TypeError(cmp.args[i].meta.replace(parent=cmp.meta), f"Compare not implemented for type {inst.type}")


    op = {
        '==': '=',
        '!in': 'not in',    # TODO !in
        '<>': '!=',
    }.get(cmp.op, cmp.op)

    code = sql_cls(op, [i.code for i in insts])
    inst = objects.Instance.make(code, types.Bool, insts)
    return inst


@dy
def compile_remote(state: State, arith: ast.Arith):
    args = evaluate(state, arith.args)

    for i, a in enumerate(args):
        if a.type.actual_type() is types.null:
            raise pql_TypeError(arith.args[i].meta.replace(parent=arith.meta), "Cannot perform arithmetic on null values")

    return _compile_arith(state, arith, *args)

@dy
def _compile_arith(state, arith, a: objects.TableInstance, b: objects.TableInstance):
    # TODO validate types
    ops = {
        "+": 'concat',
        "&": 'intersect',
        "|": 'union',
        "-": 'substract',
    }
    # TODO compile preql funccall?
    try:
        op = ops[arith.op]
    except KeyError:
        meta = arith.op.meta.replace(parent=arith.meta)
        raise pql_TypeError(meta, f"Operation '{arith.op}' not supported for tables")

    try:
        return state.get_var(op).func(state, a, b)
        # return call_pql_func(state, op, [a,b])
    except PreqlError as e:
        raise e.replace(meta=arith.meta) from e


@dy
def _compile_arith(state, arith, a, b):
    args = [a, b]
    arg_types = [a.type.actual_type() for a in args]
    arg_types_set = set(arg_types) - {types.ListType(types.any_t)}  # XXX hacky

    if settings.optimize:
        if isinstance(args[0], objects.ValueInstance) and isinstance(args[1], objects.ValueInstance):
            # Local folding for better performance (optional, for better performance)
            v1, v2 = [a.local_value for a in args]
            if arith.op == '+' and len(arg_types_set) == 1:
                return new_value_instance(v1 + v2, args[0].type)

    if len(arg_types_set) > 1:
        # Auto-convert int+float into float
        # TODO use dispatch+operator_overload+SQL() to do this in preql instead of here?
        if arg_types_set == {types.Int, types.Float}:
            arg_types_set = {types.Float}
        elif arg_types_set == {types.Int, types.String}:
            if arith.op != '*':
                meta = arith.op.meta.replace(parent=arith.meta)
                raise pql_TypeError(meta, f"Operator '{arith.op}' not supported between string and integer.")

            # REPEAT(str, int) -> str
            ordered_args = {
                (types.String, types.Int): args,
                (types.Int, types.String): args[::-1],
            }[tuple(arg_types)]

            return call_pql_func(state, "repeat", ordered_args)
        else:
            meta = arith.op.meta.replace(parent=arith.meta)
            raise pql_TypeError(meta, f"All values provided to '{arith.op}' must be of the same type (got: {arg_types})")

    # TODO check instance type? Right now ColumnInstance & ColumnType make it awkward

    if not all(isinstance(a.type.actual_type(), (types.Primitive, types.Aggregated)) for a in args):
        meta = arith.op.meta.replace(parent=arith.meta)
        raise pql_TypeError(meta, f"Operation {arith.op} not supported for type: {args[0].type, args[1].type}")

    res_type ,= {a.type.actual_type() for a in args}
    code = sql.arith(res_type, arith.op, [a.code for a in args], arith.meta)
    return objects.make_instance(code, res_type, args)


@dy
def compile_remote(state: State, x: ast.Ellipsis):
    raise pql_SyntaxError(x.meta, "Ellipsis not allowed here")


@dy
def compile_remote(state: State, c: ast.Const):
    if c.type == types.null:
        assert c.value is None
        return objects.null
    return new_value_instance(c.value, c.type)

@dy
def compile_remote(state: State, d: ast.Dict_):
    elems = {k:evaluate(state, objects.from_python(v)) for k,v in d.elems.items()}
    t = types.TableType('_dict', SafeDict({k:v.type for k,v in elems.items()}), False, [])
    code = sql.RowDict({k:v.code for k,v in elems.items()})
    return objects.ValueInstance.make(code, types.RowType(t), [], d.elems)

@dy
def compile_remote(state: State, lst: ast.List_):
    # TODO generate (a,b,c) syntax for IN operations, with its own type
    # sql = "(" * join([e.code.text for e in objs], ",") * ")"
    # type = length(objs)>0 ? objs[1].type : nothing
    # return Instance(Sql(sql), ArrayType(type, false))
    # Or just evaluate?

    if not lst.elems and lst.type.elemtype is types.any_t:
        # XXX a little awkward
        return objects.EmptyList

    elems = evaluate(state, lst.elems)

    type_set = {e.type for e in elems}
    if len(type_set) > 1:
        raise pql_TypeError(lst.meta, "Cannot create a list of mixed types: (%s)" % ', '.join(repr(t) for t in type_set))
    elif type_set:
        elem_type ,= type_set
    else:
        elem_type = lst.type.elemtype

    # XXX should work with a better type system where isa(int, any) == true
    # assert isinstance(elem_type, type(lst.type.elemtype)), (elem_type, lst.type.elemtype)


    # code = sql.TableArith(table_type, 'UNION ALL', [ sql.SelectValue(e.type, e.code) for e in elems ])
    list_type = types.ListType(elem_type)
    name = state.unique_name("list_")
    table_code, subq = sql.create_list(list_type, name, [e.code for e in elems])

    inst = objects.TableInstance.make(table_code, list_type, elems)
    inst.subqueries[name] = subq
    return inst


@dy
def compile_remote(state: State, s: ast.Slice):
    table = evaluate(state, s.table)
    # TODO if isinstance(table, objects.Instance) and isinstance(table.type, types.String):

    assert_type(s.meta, table.type, types.Collection, "Slice expected an object of type '%s', instead got '%s'")

    instances = [table]
    if s.range.start:
        start = evaluate(state, s.range.start)
        instances += [start]
    else:
        start = new_value_instance(0)

    if s.range.stop:
        stop = evaluate(state, s.range.stop)
        instances += [stop]
    else:
        stop = None

    code = sql.table_slice(table, start.code, stop and stop.code)
    # return table.remake(code=code)
    return objects.TableInstance.make(code, table.type, instances)

@dy
def compile_remote(state: State, sel: ast.Selection):
    table = evaluate(state, sel.table)
    if isinstance(table, types.PqlType):
        return _apply_type_generics(state, table, sel.conds)

    if not isinstance(table, objects.Instance):
        return sel.replace(table=table)

    assert_type(sel.meta, table.type, types.TableType, "Selection expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.all_attrs()):
        conds = evaluate(state, sel.conds)

    code = sql.table_selection(table, [c.code for c in conds])

    return objects.TableInstance.make(code, table.type, [table] + conds)

@dy
def compile_remote(state: State, param: ast.Parameter):
    if state.access_level == state.AccessLevels.COMPILE:
        return objects.Instance.make(sql.Parameter(param.type, param.name), param.type, [])
    else:
        return state.get_var(param.name)


def _apply_type_generics(state, gen_type, type_names):
    type_objs = evaluate(state, type_names)
    if not type_objs:
        raise pql_TypeError(None, f"Generics expression expected a type, got nothing.")
    for o in type_objs:
        if not isinstance(o, types.PqlType):
            raise pql_TypeError(None, f"Generics expression expected a type, got '{o}'.")

    if len(type_objs) > 1:
        #t = types.Union
        raise pql_TypeError("Union types not yet supported!")
    else:
        t ,= type_objs

    try:
        return gen_type.apply_inner_type(t)
    except TypeError:
        raise pql_TypeError(None, f"Type {t} isn't a container!")



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




# def rename_field(state, table, old, new):
#     proj = ast.Projection(None, table, [
#         ast.NamedField(None, None, ast.Ellipsis(None, [])),
#         ast.NamedField(None, new, ast.Name(None, old))
#         ])
#     return evaluate(state, proj)