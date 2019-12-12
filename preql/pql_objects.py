"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable

from .utils import dataclass, SafeDict, safezip, split_at_index
from .exceptions import pql_TypeError, pql_AttributeError
from . import pql_types as types
from . import pql_ast as ast
from .sql import Sql, RawSql


# Functions
@dataclass
class Param(ast.Ast):
    name: str
    # _type: PqlType = None

class Function(types.PqlObject):
    param_collector = None

    def match_params(self, args):
        # TODO Default values (maybe just initialize match_params with them?)

        # Canonize args for the rest of the function
        args = [a if isinstance(a, ast.NamedField) else ast.NamedField(None, None, a) for a in args]

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
    code: Sql
    type: types.PqlType

    subqueries: SafeDict

    @classmethod
    def make(cls, code, type_, instances, *extra):
        return cls(code, type_, merge_subqueries(instances), *extra)

    def get_attr(self, name):
        raise NotImplementedError(f"get_attr() not implemented for instance of type {self.type} -- code=({self.code})")

    # def __created__(self):
    #     assert self.code.type.concrete_type() == self.type.concrete_type(), (self.code.type, self.type)


@dataclass
class ColumnInstance(Instance):
    type: types.ColumnType


class DatumColumnInstance(ColumnInstance):
    def flatten(self):
        return [self]

    def remake(self, code):
        return type(self)(code, self.type, self.subqueries)

@dataclass
class StructColumnInstance(ColumnInstance):
    members: dict

    def flatten(self):
        return [atom for m in self.members.values() for atom in m.flatten()]

    def get_attr(self, name):
        return self.members[name]


def make_column_instance(code, type_, from_instances=[]):
    kernel = type_.kernel_type()

    if isinstance(kernel, types.StructColumnType):
        struct_sql_name = code.compile().text
        members = {name: make_column_instance(RawSql(member.type, struct_sql_name+'_'+name), member)
                   for name, member in kernel.members.items()}
        return StructColumnInstance.make(code, type_, from_instances, members)
    else:
        return DatumColumnInstance.make(code, type_, from_instances)
    assert False, type_

def make_instance(code, type_, from_instances=[]):
    if isinstance(type_, types.ColumnType):
        return make_column_instance(code, type_, from_instances)
    # elif isinstance(type_, Aggregated) and isinstance(type_.elemtype, ColumnType):
    #     return make_column_instance(code, type_, from_instances)
    # elif isinstance(type_, TableType):
    #     return instanciate_table(state, t, sql.TableName(t, t.name), [])

    return Instance.make(code, type_, from_instances)

def make_value_instance(value, type_):
    from .interp_common import sql_repr # XXX
    assert isinstance(type_, types.Primitive)
    return ValueInstance.make(sql_repr(value), type_, [], value)


@dataclass
class ValueInstance(Instance):
    local_value: object


@dataclass
class TableInstance(Instance):
    columns: dict

    def remake(self, code):
        return type(self)(code, self.type, self.subqueries, self.columns)

    def get_attr(self, name):
        try:
            return ColumnInstanceWithTable(self.columns[name], self)
        except KeyError:
            raise pql_AttributeError(name.meta, name)

    def flatten(self):
        return [atom for col in self.columns.values() for atom in col.flatten()]


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
    def flatten(self):
        return self.column.flatten()
    def get_attr(self, name):
        return self.column.get_attr(name)


def merge_subqueries(instances):
    return SafeDict().update(*[i.subqueries for i in instances])


def aggregated(inst):
    if isinstance(inst, TableInstance):
        new_cols = {name:aggregated(c) for name, c in inst.columns.items()}
        return TableInstance.make(inst.code, types.Aggregated(inst.type), [inst], new_cols)

    elif isinstance(inst, ColumnInstance):
        return make_column_instance(inst.code, types.make_column(inst.type.name, types.Aggregated(inst.type.type)), [inst])

    assert not isinstance(inst.type, types.TableType), inst.type
    return make_instance(inst.code, types.Aggregated(inst.type), [inst])


@dataclass
class InstancePlaceholder:
    "Half instance, half type"

    type: types.TableType

    def concrete_type(self):
        return self.type.concrete_type()

    def kernel_type(self):
        return self.type.kernel_type()