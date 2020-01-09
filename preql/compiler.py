from .utils import safezip, listgen, SafeDict
from .exceptions import pql_TypeError, PreqlError, pql_AttributeError, pql_SyntaxError

from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import dy, State, get_alias, simplify, assert_type, GlobalSettings, make_value_instance

Sql = sql.Sql

@dy
def compile_type_def(state: State, table: types.TableType) -> Sql:
    posts = []
    pks = []
    columns = []

    for path, c in table.flatten():
        if c.is_concrete:
            type_ = compile_type(state, c)
            name = "_".join(path)
            columns.append( f"{name} {type_}" )
            if isinstance(c, types.RelationalColumn):
                # TODO any column, using projection / get_attr
                if not table.temporary:
                    # In postgres, constraints on temporary tables may reference only temporary tables
                    s = f"FOREIGN KEY({name}) REFERENCES {c.type.name}(id)"
                    posts.append(s)
            if c.primary_key:
                pks.append(name)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    if table.temporary:
        return sql.RawSql(types.null, f"CREATE TEMPORARY TABLE {table.name} (" + ", ".join(columns + posts) + ")")
    else:
        return sql.RawSql(types.null, f"CREATE TABLE if not exists {table.name} (" + ", ".join(columns + posts) + ")")

@dy
def compile_type(state: State, type_: types.RelationalColumn):
    # TODO might have a different type
    return 'INTEGER'    # Foreign-key is integer

@dy
def compile_type(state: State, type_: types.DatumColumn):
    return compile_type(state, type_.type)

@dy
def compile_type(state: State, type: types.Primitive):
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'datetime': "TIMESTAMP",
    }[type.name]
    if not type.nullable:
        s += " NOT NULL"
    return s

@dy
def compile_type(state: State, idtype: types.IdType):
    if state.db.target == sql.postgres:
        return "SERIAL" # Postgres
    else:
        return "INTEGER"


def _process_fields(state: State, fields):
    "Returns {var_name: (col_instance, sql_alias)}"
    processed_fields = []
    for f in fields:

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name
        sql_friendly_name = name.replace(".", "_")

        v = compile_remote(state, f.value)

        if isinstance(v.type, types.Aggregated):

            if isinstance(v, objects.StructColumnInstance):
                raise NotImplementedError("Cannot make an array of structs at the moment.")

            v = objects.make_column_instance(sql.MakeArray(v.type, v.code), v.type, [v])

        v = _ensure_col_instance(state, f.meta, v)

        processed_fields.append( [name, (v, get_alias(state, sql_friendly_name)) ] )   # TODO Don't create new alias for fields that don't change?

    return processed_fields

def _ensure_col_instance(state, meta, i):
    if isinstance(i, objects.ColumnInstance):
        return i
    elif isinstance(i, objects.Instance):
        if isinstance(i.type, (types.Primitive, types.NullType)):
            return objects.make_column_instance(i.code, i.type, [i])

    raise pql_TypeError(meta, f"Expected a valid expression. Instead got: {i.repr(state)}")
    # assert False, i



@listgen
def _expand_ellipsis(table, fields):
    direct_names = [f.value.name for f in fields if isinstance(f.value, ast.Name)]

    for f in fields:
        assert isinstance(f, ast.NamedField)

        if not isinstance(f.value, ast.Ellipsis):
            yield f
            continue

        if f.name:
            raise pql_SyntaxError(f.meta, "Cannot use a name for ellipsis (inlining operation doesn't accept a name)")
        else:
            for name in table.columns:
                if name not in direct_names:
                    yield ast.NamedField(f.meta, name, ast.Name(None, name))


@dy
def compile_remote(state: State, proj: ast.Projection):
    table = compile_remote(state, proj.table)
    assert_type(proj.meta, table.type, (types.TableType, types.ListType, types.StructType), "Projection expected an object of type '%s', instead got '%s'")
    assert isinstance(table, (objects.TableInstance, objects.StructColumnInstance)), table

    fields = _expand_ellipsis(table, proj.fields)

    used = set()
    for f in list(proj.fields) + list(proj.agg_fields):
        if f.name:  # Otherwise, an automatic name is used, and collision is impossible (should be impossible)
            if f.name in used:
                raise pql_TypeError(f.meta, f"Field '{f.name}' was already used in this projection")
            used.add(f.name)

    columns = table.members if isinstance(table, objects.StructColumnInstance) else table.columns

    with state.use_scope(columns):
        fields = _process_fields(state, fields)

    agg_fields = []
    if proj.agg_fields:
        with state.use_scope({n:objects.aggregated(c) for n,c in columns.items()}):
            agg_fields = _process_fields(state, proj.agg_fields)


    if isinstance(table, objects.StructColumnInstance):
        assert not agg_fields
        members = {name: inst for name, (inst, _a) in fields}
        struct_type = types.StructType(get_alias(state, "struct_proj"), {name:m.type for name, m in members.items()})
        return objects.StructColumnInstance.make(table.code, struct_type, [], members)


    # Make new type
    all_aliases = []
    new_columns = {}
    new_table_type = types.TableType(get_alias(state, table.type.name + "_proj"), SafeDict(), True)
    for name_, (remote_col, sql_alias) in fields + agg_fields:
        # TODO what happens if automatic name preceeds and collides with user-given name?
        name = name_
        i = 1
        while name in new_columns:
            name = name_ + str(i)
            i += 1
        new_table_type.columns[name] = remote_col.type
        ci = objects.make_column_instance(sql.Name(remote_col.type, sql_alias), remote_col.type, [remote_col])
        new_columns[name] = ci
        all_aliases.append((remote_col, ci))

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in all_aliases
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    groupby = []
    if proj.groupby and fields:
        groupby = [sql.Name(rc.type, sql_alias) for _n, (rc, sql_alias) in fields]

    code = sql.Select(new_table_type, table.code, sql_fields, group_by=groupby)

    # Make Instance
    inst = objects.TableInstance.make(code, new_table_type, [table], new_columns)

    return inst
    # return add_as_subquery(state, inst)


@dy
def compile_remote(state: State, order: ast.Update):
    return compile_remote(state, simplify(state, order))

@dy
def compile_remote(state: State, order: ast.Order):
    table = compile_remote(state, order.table)
    assert_type(order.meta, table.type, types.TableType, "'order' expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.columns):
        fields = compile_remote(state, order.fields)

    fields = [_ensure_col_instance(state, of.meta, f) for f, of in safezip(fields, order.fields)]

    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], order=[c.code for c in fields])

    return objects.TableInstance.make(code, table.type, [table] + fields, table.columns)

@dy
def compile_remote(state: State, expr: ast.DescOrder):
    obj = compile_remote(state, expr.value)
    return obj.remake(code=sql.Desc(obj.type, obj.code))



@dy
def compile_remote(state: State, lst: list):
    return [compile_remote(state, e) for e in lst]


@dy
def compile_remote(state: State, like: ast.Like):
    s = compile_remote(state, like.str)
    p = compile_remote(state, like.pattern)
    if s.type != types.String:
        raise pql_TypeError(like.str.meta.remake(parent=like.meta), f"Like (~) operator expects two strings")
    if p.type != types.String:
        raise pql_TypeError(like.pattern.meta.remake(parent=like.meta), f"Like (~) operator expects two strings")

    code = sql.Like(types.Bool, s.code, p.code)
    return objects.Instance.make(code, types.Bool, [s, p])

@dy
def compile_remote(state: State, cmp: ast.Compare):
    insts = compile_remote(state, cmp.args)

    if cmp.op == 'in' or cmp.op == '^in':
        sql_cls = sql.Contains
        assert_type(cmp.meta, insts[0].type, types.AtomicType, "Expecting type %s, got %s")
        assert_type(cmp.meta, insts[1].type, types.Collection, "Expecting type %s, got %s")
        cols = insts[1].columns
        if len(cols) > 1:
            raise pql_TypeError(cmp.meta, "Contains operator expects a collection with only 1 column! (Got %d)" % len(cols))
        if list(cols.values())[0].type != insts[0].type:
            raise pql_TypeError(cmp.meta, "Contains operator expects all types to match")

    else:
        sql_cls = sql.Compare
        for i in insts:
            assert_type(cmp.meta, i.type, types.AtomicType, "Expecting type %s, got %s")
        # TODO should be able to coalesce, int->float, id->int, etc.
        #      also different types should still be comparable to some degree?
        # type_set = {i.type for i in insts}
        # if len(type_set) > 1:
        #     raise pql_TypeError(cmp.meta, "Cannot compare two different types: %s" % type_set)

    op = {
        '==': '=',
        '^in': 'not in'
    }.get(cmp.op, cmp.op)

    code = sql_cls(types.Bool, op, [e.code for e in insts])
    inst = objects.Instance.make(code, types.Bool, insts)
    return inst

@dy
def compile_remote(state: State, attr: ast.Attr):
    inst = compile_remote(state, attr.expr)

    # if isinstance(inst, types.PqlType): # ugly
    #     if attr.name == '__name__':
    #         return compile_remote(state, ast.Const(None, types.String, str(inst.name)))
    #     raise pql_AttributeError(attr.meta, "'%s' has no attribute '%s'" % (inst, attr.name))

    try:
        return inst.get_attr(attr.name)
    except pql_AttributeError:
        meta = attr.name.meta.remake(parent=attr.meta)
        raise pql_AttributeError(meta, f"Objects of type '{inst.type}' have no attributes (for now)")



def call_pql_func(state, name, args):
    expr = ast.FuncCall(None, ast.Name(None, name), args)
    return compile_remote(state, expr)

@dy
def compile_remote(state: State, arith: ast.Arith):
    args = compile_remote(state, arith.args)
    arg_types = [a.type for a in args]
    arg_types_set = set(arg_types)

    if GlobalSettings.Optimize:
        if isinstance(args[0], objects.ValueInstance) and isinstance(args[1], objects.ValueInstance):
            # Local folding for better performance (optional, for better performance)
            v1, v2 = [a.local_value for a in args]
            if arith.op == '+' and len(arg_types_set) == 1:
                return make_value_instance(v1 + v2, args[0].type)

    if len(arg_types_set) > 1:
        # Auto-convert int+float into float
        # TODO use dispatch+operator_overload+SQL() to do this in preql instead of here?
        if arg_types_set == {types.Int, types.Float}:
            arg_types_set = {types.Float}
        elif arg_types_set == {types.Int, types.String}:
            if arith.op != '*':
                meta = arith.op.meta.remake(parent=arith.meta)
                raise pql_TypeError(meta, f"Operator '{arith.op}' not supported between string and integer.")

            # REPEAT(str, int) -> str
            if arg_types == [types.String, types.Int]:
                ordered_args = args
            elif arg_types == [types.Int, types.String]:
                ordered_args = args[::-1]
            else:
                assert False

            # expr = ast.FuncCall(None, ast.Name(None, "repeat"), ordered_args)
            # return compile_remote(state, expr)
            return call_pql_func(state, "repeat", ordered_args)
        else:
            meta = arith.op.meta.remake(parent=arith.meta)
            raise pql_TypeError(meta, f"All values provided to '{arith.op}' must be of the same type (got: {arg_types})")

    # TODO check instance type? Right now ColumnInstance & ColumnType make it awkward

    if isinstance(args[0], objects.TableInstance):
        assert isinstance(args[1], objects.TableInstance)
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
            meta = arith.op.meta.remake(parent=arith.meta)
            raise pql_TypeError(meta, f"Operation '{arith.op}' not supported for tables")

        return state.get_var(op).func(state, *args)

    if not all(isinstance(a.type, (types.Primitive, types.ListType)) for a in args):
        meta = arith.op.meta.remake(parent=arith.meta)
        raise pql_TypeError(meta, f"Operation {arith.op} not supported for type: {args[0].type, args[1].type}")

    arg_codes = [a.code for a in args]
    res_type ,= arg_types_set
    op = arith.op
    if arg_types_set == {types.String}:
        if arith.op != '+':
            meta = arith.op.meta.remake(parent=arith.meta)
            raise pql_TypeError(meta, f"Operator '{arith.op}' not supported for strings.")
        op = '||'
    elif arith.op == '/':
        arg_codes[0] = sql.Cast(types.Float, 'float', arg_codes[0])
        arg_types = types.Float
    elif arith.op == '//':
        op = '/'

    code = sql.Arith(res_type, op, arg_codes)
    return objects.Instance.make(code, res_type, args)


@dy # Could happen because sometimes simplify(...) calls compile_remote
def compile_remote(state: State, i: objects.Instance):
    return i
@dy
def compile_remote(state: State, i: objects.TableInstance):
    return i
@dy
def compile_remote(state: State, f: ast.FuncCall):
    res = simplify(state, f)
    return compile_remote(state, res)
@dy
def compile_remote(state: State, f: objects.UserFunction):
    "Functions don't need compilation"
    return f
@dy
def compile_remote(state: State, f: objects.InternalFunction):
    "Functions don't need compilation"
    return f
@dy
def compile_remote(state: State, x: ast.Ellipsis):
    raise pql_SyntaxError(x.meta, "Ellipsis not allowed here")


@dy
def compile_remote(state: State, c: ast.Const):
    return make_value_instance(c.value, c.type)

@dy
def compile_remote(state: State, n: ast.Name):
    v = simplify(state, n)
    assert v is not n   # Protect against recursions
    return compile_remote(state, v)

@dy
def compile_remote(state: State, lst: objects.List_):
    # TODO generate (a,b,c) syntax for IN operations, with its own type
    # sql = "(" * join([e.code.text for e in objs], ",") * ")"
    # type = length(objs)>0 ? objs[1].type : nothing
    # return Instance(Sql(sql), ArrayType(type, false))
    # Or just evaluate?

    if not lst.elems:
        list_type = types.ListType(types.null)      # XXX Any type?
        code = sql.EmptyList(list_type)
        return instanciate_table(state, list_type, code, [])

    elems = compile_remote(state, lst.elems)

    type_set = list({e.type for e in elems})
    if len(type_set) > 1:
        raise pql_TypeError(lst.meta, "Cannot create a list of mixed types: (%s)" % ', '.join(repr(t) for t in type_set))
    else:
        elem_type ,= type_set

    list_type = types.ListType(elem_type)

    # code = sql.TableArith(table_type, 'UNION ALL', [ sql.SelectValue(e.type, e.code) for e in elems ])
    name = get_alias(state, "list_")
    inst = instanciate_table(state, list_type, sql.TableName(list_type, name), elems)
    fields = [sql.Name(elem_type, 'value')]
    subq = sql.Subquery(list_type, name, fields, sql.Values(list_type, [e.code for e in elems]))
    inst.subqueries[name] = subq
    return inst


@dy
def compile_remote(state: State, t: types.TableType):
    return t
@dy
def compile_remote(state: State, t: types.FunctionType):
    return t


@dy
def compile_remote(state: State, s: ast.Slice):
    table = compile_remote(state, s.table)
    # TODO if isinstance(table, objects.Instance) and isinstance(table.type, types.String):

    assert_type(s.meta, table.type, types.Collection, "Slice expected an object of type '%s', instead got '%s'")

    instances = [table]
    if s.range.start:
        start = compile_remote(state, s.range.start)
        instances += [start]
    else:
        start = make_value_instance(0)
    if s.range.stop:
        stop = compile_remote(state, s.range.stop)
        instances += [stop]
        limit = sql.Arith(types.Int, '-', [stop.code, start.code])
    else:
        limit = None

    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], offset=start.code, limit=limit)
    # return table.remake(code=code)
    return objects.TableInstance.make(code, table.type, instances, table.columns)

@dy
def compile_remote(state: State, sel: ast.Selection):
    table = compile_remote(state, sel.table)
    assert_type(sel.meta, table.type, types.TableType, "Selection expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.columns):
        conds = compile_remote(state, sel.conds)

    conds = [_ensure_col_instance(state, of.meta, f) for f, of in safezip(conds, sel.conds)]

    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], conds=[c.code for c in conds])
    inst = objects.TableInstance.make(code, table.type, [table] + conds, table.columns)
    return inst



@dy
def compile_remote(state: State, obj: objects.StructColumnInstance):
    return obj
@dy
def compile_remote(state: State, obj: objects.DatumColumnInstance):
    return obj
@dy
def compile_remote(state: State, obj: types.Primitive):
    return obj
@dy
def compile_remote(state: State, obj: types.ListType):
    return obj
@dy
def compile_remote(state: State, obj: objects.ValueInstance):
    return obj


def guess_field_name(f):
    if isinstance(f, ast.Attr):
        return guess_field_name(f.expr) + "." + f.name
    elif isinstance(f, ast.Name):
        return str(f.name)
    elif isinstance(f, ast.Projection): # a restructre within a projection
        return guess_field_name(f.table)
    elif isinstance(f, ast.FuncCall):
        return guess_field_name(f.func)
    return '_'



def instanciate_column(state: State, name, t):
    return objects.make_column_instance(sql.Name(t, get_alias(state, name)), t)


def _make_name(parts):
    return '_'.join(parts)


def alias_table(state: State, t):
    new_columns = {
        name: objects.make_column_instance(sql.Name(col.type, get_alias(state, name)), col.type, [col])
        for name, col in t.columns.items()
    }

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in safezip(t.columns.values(), new_columns.values())
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    code = sql.Select(t.type, t.code, sql_fields)
    return t.remake(code=code, columns=new_columns)


def instanciate_table(state: State, t: types.TableType, source: Sql, instances, values=None):
    if values is None:
        columns = {name: objects.make_column_instance(sql.Name(c, name), c) for name, c in t.columns.items()}
        return objects.TableInstance(source, t, objects.merge_subqueries(instances), columns)

    columns = {name: instanciate_column(state, name, c) for name, c in t.columns.items()}

    atoms = [atom
                for name, inst in columns.items()
                for atom in inst.flatten_path([name])
            ]

    aliases = [ sql.ColumnAlias.make(value, atom.code) for value, (_, atom) in safezip(values, atoms) ]

    code = sql.Select(t, source, aliases)

    return objects.TableInstance(code, t, objects.merge_subqueries(instances), columns)