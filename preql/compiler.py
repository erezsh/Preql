from .utils import safezip
from .exceptions import pql_TypeError, PreqlError, pql_AttributeError

from . import pql_types as types
from . import pql_objects as objects
from . import pql_ast as ast
from . import sql
from .interp_common import dy, State, get_alias, simplify, assert_type, sql_repr

Sql = sql.Sql

@dy
def compile_type_def(table: types.TableType) -> Sql:
    posts = []
    pks = []
    columns = []

    for c in table.flatten():
        if c.is_concrete:
            type_ = compile_type(c.type)
            columns.append( f"{c.name} {type_}" )
            if isinstance(c, types.RelationalColumnType):
                # TODO any column, using projection / get_attr
                s = f"FOREIGN KEY({c.name}) REFERENCES {c.type.name}(id)"
                posts.append(s)
            if c.primary_key:
                pks.append(c.name)

    if pks:
        names = ", ".join(pks)
        posts.append(f"PRIMARY KEY ({names})")

    # Consistent among SQL databases
    if table.temporary:
        return sql.RawSql(types.null, f"CREATE TEMPORARY TABLE {table.name} (" + ", ".join(columns + posts) + ")")
    else:
        return sql.RawSql(types.null, f"CREATE TABLE if not exists {table.name} (" + ", ".join(columns + posts) + ")")

@dy
def compile_type(type_: types.TableType):
    return type_.name

@dy
def compile_type(type: types.Primitive):
    s = {
        'int': "INTEGER",
        'string': "VARCHAR(4000)",
        'float': "FLOAT",
        'bool': "BOOLEAN",
        'date': "TIMESTAMP",
    }[type.name]
    if not type.nullable:
        s += " NOT NULL"
    return s




def _process_fields(state: State, fields):
    "Returns {var_name: (col_instance, sql_alias)}"
    processed_fields = {}
    for f in fields:
        if f.name in processed_fields:
            raise pql_TypeError(f.meta, f"Field '{f.name}' was already used in this projection")

        suggested_name = str(f.name) if f.name else guess_field_name(f.value)
        name = suggested_name.rsplit('.', 1)[-1]    # Use the last attribute as name
        sql_friendly_name = name.replace(".", "_")
        unique_name = name
        i = 1
        while unique_name in processed_fields:
            i += 1
            unique_name = name + str(i)

        v = compile_remote(state, f.value)


        # TODO move to ColInstance?
        if isinstance(v, objects.TableInstance):
            t = types.make_column(name, types.StructType(v.type.name, {n:c.type.type for n,c in v.columns.items()}))
            v = objects.StructColumnInstance(v.code, t, v.subqueries, v.columns)

        # SQL uses `first` by default on aggregate columns. This will force SQL to create an array by default.
        # TODO first() method to take advantage of this ability (although it's possible with window functions too)
        concrete_type = v.type.concrete_type()
        if isinstance(concrete_type, types.Aggregated):
            col_type = types.make_column(get_alias(state, "list"), concrete_type)
            v = objects.make_column_instance(sql.MakeArray(concrete_type, v.code), col_type, [v])

        v = _ensure_col_instance(v)

        processed_fields[unique_name] = v, get_alias(state, sql_friendly_name)   # TODO Don't create new alias for fields that don't change?

    return list(processed_fields.items())

def _ensure_col_instance(i):
    if isinstance(i, objects.ColumnInstance):
        return i
    elif isinstance(i, objects.Instance):
        if isinstance(i.type, types.Primitive):
            return objects.make_column_instance(i.code, types.make_column('_anonymous_instance', i.type))

    assert False, i




@dy
def compile_remote(state: State, proj: ast.Projection):
    table = compile_remote(state, proj.table)
    assert_type(table.type, (types.TableType, types.ListType), "Projection expected an object of type '%s', instead got '%s'")
    assert isinstance(table, objects.TableInstance), table

    columns = table.members if isinstance(table, objects.StructColumnInstance) else table.columns

    with state.use_scope(columns):
        fields = _process_fields(state, proj.fields)

    agg_fields = []
    if proj.agg_fields:
        with state.use_scope({n:objects.aggregated(c) for n,c in columns.items()}):
            agg_fields = _process_fields(state, proj.agg_fields)

    if isinstance(table, objects.StructColumnInstance):
        members = {name: inst.type for name, (inst, _a) in fields + agg_fields}
        struct_type = types.StructType(get_alias(state, table.type.name + "_proj"), members)
        struct_col_type = types.make_column("<this is meaningless?>", struct_type)
        return objects.make_column_instance(table.code, struct_col_type, [table])

    # Make new type
    all_aliases = []
    new_table_type = types.TableType(get_alias(state, table.type.name + "_proj"), {}, True)
    for name, (remote_col, sql_alias) in fields + agg_fields:
        new_col_type = remote_col.type.remake(name=name)
        new_table_type.add_column(new_col_type)
        ci = objects.make_column_instance(sql.Name(new_col_type.type, sql_alias), new_col_type, [remote_col])
        all_aliases.append((remote_col, ci))

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in all_aliases
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    groupby = []
    if proj.groupby and fields:
        groupby = [sql.Name(rc.type.type, sql_alias) for _n, (rc, sql_alias) in fields]

    code = sql.Select(new_table_type, table.code, sql_fields, group_by=groupby)

    # Make Instance
    columns = {new.type.name:new for old, new in all_aliases}
    inst = objects.TableInstance.make(code, new_table_type, [table], columns)

    return inst
    # return add_as_subquery(state, inst)


@dy
def compile_remote(state: State, order: ast.Update):
    return compile_remote(state, simplify(state, order))

@dy
def compile_remote(state: State, order: ast.Order):
    table = compile_remote(state, order.table)
    assert_type(table.type, types.TableType, "'order' expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.columns):
        fields = compile_remote(state, order.fields)

    code = sql.Select(table.type, table.code, [sql.AllFields(table.type)], order=[c.code for c in fields])

    return objects.TableInstance.make(code, table.type, [table] + fields, table.columns)

@dy
def compile_remote(state: State, expr: ast.DescOrder):
    obj = compile_remote(state, expr.value)
    return obj.remake(code=sql.Desc(obj.type, obj.code))



@dy
def compile_remote(state: State, lst: list):
    return [compile_remote(state, e) for e in lst]

# @dy
# def compile_remote(state: State, c: ast.Contains):
#     args = compile_remote(state, c.args)
#     code = sql.Contains(types.Bool, c.op, [a.code for a in args])
#     return objects.make_instance(code, types.Bool, args)

@dy
def compile_remote(state: State, cmp: ast.Compare):
    sql_cls = sql.Compare
    if cmp.op == 'in' or cmp.op == '^in':
        sql_cls = sql.Contains

    op = {
        '==': '=',
        '^in': 'not in'
    }.get(cmp.op, cmp.op)
    code = sql_cls(types.Bool, op, [e.code for e in compile_remote(state, cmp.args)])
    return objects.Instance.make(code, types.Bool, [])

@dy
def compile_remote(state: State, attr: ast.Attr):
    inst = compile_remote(state, attr.expr)

    if isinstance(inst, types.PqlType): # XXX ugly
        if attr.name == '__name__':
            return compile_remote(state, ast.Const(None, types.String, str(inst.name)))
        raise pql_AttributeError(attr.meta, "'%s' has no attribute '%s'" % (inst, attr.name))

    return inst.get_attr(attr.name)

@dy
def compile_remote(state: State, arith: ast.Arith):
    args = compile_remote(state, arith.args)

    arg_codes = [a.code for a in args]
    arg_types = [a.type.concrete_type() for a in args]
    arg_types_set = set(arg_types)
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
                codes = arg_codes
            elif arg_types == [types.Int, types.String]:
                codes = arg_codes[::-1]
            else:
                assert False
            # code = sql.FuncCall(types.String, "REPEAT", codes)
            # XXX Sqlite3 hack for repeat
            # TODO move to pql_repeat
            code = sql.RawSql(types.String, f"replace(hex(zeroblob({codes[1].text})), '00', {codes[0].text})")
            return objects.make_instance(code, types.String, args)
        else:
            meta = meta_from_token(arith.op).remake(parent=arith.meta)
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
        return state.get_var(ops[arith.op]).func(state, *args)



    # TODO validate all args are compatiable
    res_type ,= arg_types_set
    op = arith.op
    if arg_types_set == {types.String}:
        if arith.op != '+':
            meta = meta_from_token(arith.op).remake(parent=arith.meta)
            raise pql_TypeError(meta, f"Operator '{arith.op}' not supported for strings.")
        op = '||'
    elif arith.op == '/':
        arg_codes[0] = sql.Cast(types.Float, 'float', arg_codes[0])
        arg_types = types.Float
    elif arith.op == '//':
        op = '/'

    code = sql.Arith(res_type, op, arg_codes)
    return objects.make_instance(code, res_type, args)


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
def compile_remote(state: State, c: ast.Const):
    return objects.Instance.make(sql_repr(c.value), c.type, [])

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

    elems = [compile_remote(state, e) for e in lst.elems]

    elem_type = elems[0].type if elems else types.PqlType
    table_type = types.ListType(elem_type)
    # table_type = types.TableType("array_%s" % elem_type.name, {}, temporary=True)   # Not temporary: ephemeral
    # table_type.add_column(types.DatumColumnType("value", elem_type))

    # TODO use values + subqueries instead of union all (better performance)
    # e.g. with list(value) as (values(1),(2),(3)) select value from list;
    code = sql.TableArith(table_type, 'UNION ALL', [ sql.SelectValue(e.type, e.code) for e in elems ])
    inst = instanciate_table(state, table_type, code, elems)
    return inst

@dy
def compile_remote(state: State, t: types.TableType):
    return t

@dy
def compile_remote(state: State, t: objects.InstancePlaceholder):
    t = t.type
    return instanciate_table(state, t, sql.TableName(t, t.name), [])



@dy
def compile_remote(state: State, sel: ast.Selection):
    table = compile_remote(state, sel.table)
    assert_type(table.type, types.TableType, "Selection expected an object of type '%s', instead got '%s'")

    with state.use_scope(table.columns):
        conds = compile_remote(state, sel.conds)

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



def instanciate_column(state: State, c: types.ColumnType):
    return objects.make_column_instance(sql.RawSql(c.type, get_alias(state, c.name)), c)

def instanciate_table(state: State, t: types.TableType, source: Sql, instances, values=None):
    columns = {name: instanciate_column(state, c) for name, c in t.columns.items()}

    atoms = [atom
                for inst in columns.values()
                for atom in inst.flatten()
            ]

    if values is None:
        values = [sql.Name(atom.type, atom.type.name) for atom in atoms]    # The column value

    aliases = [ sql.ColumnAlias.make(value, atom.code) for value, atom in safezip(values, atoms) ]

    code = sql.Select(t, source, aliases)

    return objects.TableInstance(code, t, objects.merge_subqueries(instances), columns)