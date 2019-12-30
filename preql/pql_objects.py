"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable, Any

from .utils import dataclass, SafeDict, safezip, split_at_index, concat_for
from .exceptions import pql_TypeError, pql_AttributeError
from . import pql_types as types
from . import pql_ast as ast
from . import sql

# Functions
@dataclass
class Param(ast.Ast):
    name: str
    type: Optional[Any] = None
    default: Optional[types.PqlObject] = None
    orig: Any = None # XXX temporary and lazy, for TableConstructor

class Function(types.PqlObject):
    param_collector = None

    @property
    def type(self):
        return types.FunctionType(tuple(p.type for p in self.params), self.param_collector is not None)

    def match_params(self, args):
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
            matched.append((self.param_collector, collected))
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


# Collections

@dataclass
class List_(ast.Expr):
    elems: list


# @dataclass
# class Dict_(Expr):
#     elems: dict


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
            return self.members[name]
        except KeyError:
            raise pql_AttributeError(name.meta, name)


def make_column_instance(code, type_, from_instances=()):
    kernel = type_.kernel_type()

    if isinstance(kernel, types.StructType):
        # XXX this is all wrong!
        struct_sql_name = code.compile(sql.QueryBuilder(None)).text
        members = {name: make_column_instance(sql.Name(member, struct_sql_name+'_'+name), member)
                   for name, member in kernel.members.items()}
        return StructColumnInstance.make(None, type_, from_instances, members)
    else:
        return DatumColumnInstance.make(code, type_, from_instances)
    assert False, type_


def make_value_instance(value, type_):
    from .interp_common import sql_repr, GlobalSettings # XXX
    assert isinstance(type_, types.Primitive)
    if GlobalSettings.Optimize:
        return ValueInstance.make(sql_repr(value), type_, [], value)
    else:
        return Instance.make(sql_repr(value), type_, [])


@dataclass
class ValueInstance(Instance):
    local_value: object


@dataclass
class TableInstance(Instance):
    columns: dict

    def get_attr(self, name):
        try:
            return ColumnInstanceWithTable(self.columns[name], self)
        except KeyError:
            raise pql_AttributeError(name.meta, name)

    def flatten_path(self, path=[]):
        return concat_for(col.flatten_path(path + [name]) for name, col in self.columns.items())
    def flatten(self):
        return [x for _,x in self.flatten_path()]

    def to_struct_column(self):
        # return make_column_instance(None, self.type.to_struct_type(), [self])
        return StructColumnInstance(None, self.type.to_struct_type(), self.subqueries, self.columns)


class ColumnInstanceWithTable(ColumnInstance):
    column: ColumnInstance
    table: TableInstance

    def __init__(self, column, table):
        self.column = column
        self.table = table

    @property
    def code(self):
        return self.column.code
    @property
    def type(self):
        return self.column.type
    @property
    def subqueries(self):
        return self.column.subqueries
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
        return inst.remake(type=types.Aggregated(inst.type), members=new_members)

    elif isinstance(inst, ColumnInstance):
        col_type = types.Aggregated(inst.type)
        return inst.remake(type=col_type)

    assert not isinstance(inst.type, types.TableType), inst.type
    return Instance.make(inst.code, types.Aggregated(inst.type), [inst])


null = ValueInstance.make(sql.null, types.null, [], None)