"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable, Any, Dict

from .utils import dataclass, SafeDict, safezip, split_at_index, concat_for, X
from .exceptions import pql_TypeError, pql_AttributeError
from . import settings
from . import pql_types as types
from . import pql_ast as ast
from . import sql

# Functions
@dataclass
class Param(ast.Ast):
    name: str
    type: Optional[types.PqlType] = None
    default: Optional[types.PqlObject] = None
    orig: Any = None # XXX temporary and lazy, for TableConstructor

@dataclass
class ParamDictType(types.PqlType):
    types: Dict[str, types.PqlType]

@dataclass
class ParamDict(types.PqlObject):
    params: Dict[str, types.PqlObject]

    def __len__(self):
        return len(self.params)

    def items(self):
        return self.params.items()

    @property
    def type(self):
        # XXX is ParamDictType really necessary?
        # return tuple(p.type for p in self.params.values())
        return ParamDictType({n:p.type for n, p in self.params.items()})


class Function(types.PqlObject):
    param_collector = None

    @property
    def type(self):
        return types.FunctionType(tuple(p.type for p in self.params), self.param_collector is not None)

    def match_params_fast(self, args):
        return [(p, a) for p, a in zip(self.params, args)]


    def match_params(self, args):
        # If no keyword arguments, matching is much simpler and faster
        if all(not isinstance(a, ast.NamedField) for a in args):
            return self.match_params_fast(args)

        # Canonize args for the rest of the function
        args = [a if isinstance(a, ast.NamedField) else ast.NamedField(None, None, a) for a in args]

        named = [arg.name is not None for arg in args]
        try:
            first_named = named.index(True)
        except ValueError:
            first_named = len(args)
        else:
            if not all(n for n in named[first_named:]):
                # TODO meta
                raise pql_TypeError(None, f"Function {self.name} recieved a non-named argument after a named one!")

        if first_named > len(self.params):
            # TODO meta
            raise pql_TypeError(None, f"Function '{self.name}' takes {len(self.params)} parameters but recieved {first_named} arguments.")

        values = {p.name: p.default for p in self.params}

        for pos_arg, name in zip(args[:first_named], values):
            assert pos_arg.name is None
            values[name] = pos_arg.value

        collected = {}
        if first_named is not None:
            for named_arg in args[first_named:]:
                arg_name = named_arg.name
                if arg_name in values:
                    values[arg_name] = named_arg.value
                elif self.param_collector:
                    assert arg_name not in collected
                    collected[arg_name] = named_arg.value
                else:
                    # TODO meta
                    raise pql_TypeError(None, f"Function '{self.name}' has no parameter named '{arg_name}'")


        for name, value in values.items():
            if value is None:
                # TODO meta
                raise pql_TypeError(None, f"Error calling function '{self.name}': parameter '{name}' has no value")

        matched = [(p, values.pop(p.name)) for p in self.params]
        assert not values, values
        if collected:
            matched.append((self.param_collector, ParamDict(collected)))
        return matched



    def _match_params(self, args):
        # TODO Default values (maybe just initialize match_params with them?)


        for i, arg in enumerate(args):
            if arg.name:  # First keyword argument
                pos_args, named_args = split_at_index(args, i)
                pos_params, named_params = split_at_index(self.params, i)
                break
        else:   # No keywords
            pos_args = list(args)
            pos_params = self.params
            named_args = []
            named_params = []

        if len(pos_params) != len(pos_args):
            # TODO meta
            raise pql_TypeError(None, f"Function '{self.name}' takes {len(pos_params)} parameters but recieved {len(pos_args)} arguments.")

        matched = [(p, arg.value) for (p,arg) in safezip(pos_params, pos_args)]

        args_d = {na.name: na.value for na in named_args}

        for np in named_params:
            try:
                matched.append((np, args_d.pop(np.name)))
            except KeyError:
                raise pql_TypeError("Parameter wasn't assigned: %s" % np.name)

        assert len(matched) == len(self.params)
        if args_d:  # Still left-over keywords?
            if self.param_collector:
                matched.append((self.param_collector, args_d))
            else:
                raise pql_TypeError(f"Function doesn't accept arguments named $(keys(args_d))")

        return matched



@dataclass
class UserFunction(Function):
    name: str
    params: List[Param]
    expr: (ast.Expr, ast.CodeBlock)
    param_collector: Optional[Param] = None

    def repr(self, state):
        params = ", ".join(p.name for p in self.params)
        return f'<func {self.name}({params}) ...>'

@dataclass
class InternalFunction(Function):
    name: str
    params: List[Param]
    func: Callable
    param_collector: Optional[Param] = None

    def repr(self, state):
        params = ", ".join(p.name for p in self.params)
        return f'<func {self.name}({params}) ...>'



# Other

@dataclass
class Instance(types.PqlObject):
    code: sql.Sql
    type: types.PqlType

    subqueries: SafeDict

    @classmethod
    def make(cls, code, type_, instances, *extra):
        return cls(code, type_, merge_subqueries(instances), *extra)

    def get_attr(self, name):
        raise pql_AttributeError(name.meta, name)

    def repr(self, state):
        return f'<instance of {self.type.repr(state)}>'

    def __post_init__(self):
        assert not isinstance(self.type, types.DatumColumn)



@dataclass
class ColumnInstance(Instance):
    type: types.PqlType


class DatumColumnInstance(ColumnInstance):
    def flatten_path(self, path=[]):
        return [(path, self)]
    def flatten(self):
        return [self]


@dataclass
class StructColumnInstance(ColumnInstance):
    code: type(None)
    members: dict

    def flatten_path(self, path=[]):
        return concat_for(m.flatten_path(path + [name]) for name, m in self.members.items())
    def flatten(self):
        return [x for _,x in self.flatten_path()]

    def get_attr(self, name):
        try:
            obj = self.members[name]
            assert isinstance(obj, Instance)
            return obj
        except KeyError:
            raise pql_AttributeError(name.meta, name)


def make_column_instance(code, type_, from_instances=()):
    type_ = type_.actual_type()
    kernel = type_.kernel_type()

    if isinstance(kernel, types.StructType):
        # XXX this is all wrong!
        assert isinstance(code, sql.Name)
        struct_sql_name = code.compile(sql.QueryBuilder(None)).text
        members = {name: make_column_instance(sql.Name(member, struct_sql_name+'_'+name), member)
                   for name, member in kernel.members.items()}
        return StructColumnInstance.make(None, type_, from_instances, members)
    else:
        return DatumColumnInstance.make(code, type_, from_instances)
    assert False, type_




def from_python(value):
    if value is None:
        return null
    elif isinstance(value, str):
        return ast.Const(None, types.String, value)
    elif isinstance(value, int):
        return ast.Const(None, types.Int, value)
    elif isinstance(value, list):
        # return ast.Const(None, types.ListType(types.String), value)
        return ast.List_(None, list(map(from_python, value)))
    elif isinstance(value, dict):
        return ast.Dict_(None, value)
    assert False, value


def sql_repr(x):
    if x is None:
        return sql.null

    t = types.Primitive.by_pytype[type(x)]
    if t is types.DateTime:
        # TODO Better to pass the object instead of a string?
        return sql.Primitive(t, repr(str(x)))

    if t is types.String or t is types.Text:
        return sql.Primitive(t, "'%s'" % str(x).replace("'", "''"))

    return sql.Primitive(t, repr(x))

def make_value_instance(value, type_=None, force_type=False):
    r = sql_repr(value)
    if force_type:
        assert type_
    elif type_:
        assert isinstance(type_, (types.Primitive, types.NullType, types.IdType)), type_
        assert r.type == type_, (r.type, type_)
    else:
        type_ = r.type
    if settings.optimize:   # XXX a little silly? But maybe good for tests?
        return ValueInstance.make(r, type_, [], value)
    else:
        return Instance.make(r, type_, [])


@dataclass
class ValueInstance(Instance):
    local_value: object

    def get_attr(self, name):
        if isinstance(self.type, types.RowType):
            obj = self.local_value[name]
            return from_python(obj)   # XXX Maybe use 'code' to be more efficient?
        return super().get_attr(name)



@dataclass
class TableInstance(Instance):
    columns: dict

    def get_attr(self, name):
        try:
            return ColumnReference(self, name)
        except KeyError:
            raise pql_AttributeError(name.meta, name)

    def flatten_path(self, path=[]):
        return concat_for(col.flatten_path(path + [name]) for name, col in self.columns.items())
    def flatten(self):
        return [x for _,x in self.flatten_path()]

    def to_struct_column(self):
        # return make_column_instance(None, self.type.to_struct_type(), [self])
        return StructColumnInstance(None, self.type.to_struct_type(), self.subqueries, self.columns)


@dataclass
class RowInstance(Instance):
    type: types.RowType
    table: TableInstance

    # @classmethod
    # def from_table(cls, t):
    #     rowtype = types.RowType(t.type)
    #     return cls.make(t.code.replace(type=rowtype), rowtype, [t], t)

    # def get_attr(self, name):
    #     col = table.get_attr(name)


# @dataclass
class ColumnReference(ColumnInstance):
    table: TableInstance
    name: str

    def replace(self, table):
        return type(self)(table, self.name)

    def __init__(self, table, name):
        self.table = table
        self.name = name
        assert self.column

    @property
    def column(self):
        return self.table.columns[self.name]


    code = property(X.column.code)
    type = property(X.column.type)

    @property
    def subqueries(self):
        return merge_subqueries([self.column, self.table])

    def flatten_path(self, path):
        return self.column.flatten_path(path)
    def flatten(self):
        return self.column.flatten()

    def get_attr(self, name):
        return self.column.get_attr(name)


def merge_subqueries(instances):
    return SafeDict().update(*[i.subqueries for i in instances])


def aggregated(inst):
    assert not isinstance(inst, TableInstance)  # Should be struct instead

    if isinstance(inst, StructColumnInstance):
        new_members = {name:aggregated(c) for name, c in inst.members.items()}
        # return TableInstance.make(inst.code, types.Aggregated(inst.type), [inst], new_members)
        return inst.replace(type=types.Aggregated(inst.type), members=new_members)

    elif isinstance(inst, ColumnInstance):
        col_type = types.Aggregated(inst.type)
        return inst.replace(type=col_type)

    assert not isinstance(inst.type, types.TableType), inst.type
    return Instance.make(inst.code, types.Aggregated(inst.type), [inst])


null = ValueInstance.make(sql.null, types.null, [], None)

@dataclass
class EmptyListInstance(TableInstance):
    """Special case, because it is untyped
    """
    def get_attr(self, name):
        return EmptyList

_empty_list_type = types.ListType(types.any_t)
from collections import defaultdict
def _any_column():
    return EmptyList
EmptyList = EmptyListInstance.make(sql.EmptyList(_empty_list_type), _empty_list_type, [], defaultdict(_any_column))    # Singleton


def make_instance(code, type_, from_instances=()):
    if isinstance(type_, types.Collection):
        columns = {name: make_column_instance(sql.Name(c.actual_type(), name), c) for name, c in type_.columns.items()}
        return TableInstance.make(code, type_, from_instances, columns)
    else:
        return Instance.make(code, type_, from_instances)



def instanciate_column(state, name, t, insts=[]):
    return make_column_instance(sql.Name(t, state.unique_name(name)), t, insts)


def instanciate_table(state, t: types.TableType, source: sql.Sql, instances, values=None):
    if values is None:
        columns = {name: make_column_instance(sql.Name(c.actual_type(), name), c) for name, c in t.columns.items()}
        code = source
    else:
        columns = {name: instanciate_column(state, name, c) for name, c in t.columns.items()}

        atoms = [atom
                    for name, inst in columns.items()
                    for path, atom in inst.flatten([name])
                ]

        aliases = [ sql.ColumnAlias.make(value, atom.code) for value, atom in safezip(values, atoms) ]

        code = sql.Select(t, source, aliases)

    return TableInstance(code, t, merge_subqueries(instances), columns)


def alias_table(state, t):
    new_columns = {
        name: instanciate_column(state, name, col.type, [col])
        for name, col in t.columns.items()
    }

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(o.code, n.code)
        for old, new in safezip(t.columns.values(), new_columns.values())
        for o, n in safezip(old.flatten(), new.flatten())
    ]

    code = sql.Select(t.type, t.code, sql_fields)
    return t.replace(code=code, columns=new_columns)
