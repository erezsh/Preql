"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable, Any, Dict

from .utils import dataclass, SafeDict, safezip, split_at_index, concat_for, X, listgen
from .exceptions import pql_TypeError, pql_AttributeError
from . import settings
from . import pql_types as types
from . import pql_ast as ast
from . import sql

# Functions
@dataclass
class Param(ast.Ast):
    name: str
    type: Optional[types.PqlObject] = None
    default: Optional[types.PqlObject] = None
    orig: Any = None # XXX temporary and lazy, for TableConstructor

class ParamVariadic(Param):
    pass

# @dataclass
# class ParamDictType(types.PqlType):
#     types: Dict[str, types.PqlType]

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
        return tuple((n,p.type) for p in self.params.values())
        # return ParamDictType({n:p.type for n, p in self.params.items()})


class Function(types.PqlObject):

    @property
    def type(self):
        return types.FunctionType(tuple(p.type or types.any_t for p in self.params), self.param_collector is not None)

    def help_str(self, state):
        raise NotImplementedError()

    def repr(self, state):
        return '<%s>' % self.help_str(state)


    @listgen
    def match_params_fast(self, args):
        for i, p in enumerate(self.params):
            if i < len(args):
                v = args[i]
            else:
                v = p.default
                assert v is not None

            yield p, v

        # return [(p, a) for p, a in safezip(self.params, args)]

    def _localize_keys(self, state, struct):
        raise NotImplementedError()

    def match_params(self, state, args):

        # If no keyword arguments, matching is much simpler and faster
        if all(not isinstance(a, (ast.NamedField, ast.InlineStruct)) for a in args):
            return self.match_params_fast(args)

        # Canonize args for the rest of the function
        inline_args = []
        for i, a in enumerate(args):
            if isinstance(a, ast.NamedField):
                inline_args.append(a)
            elif isinstance(a, ast.InlineStruct):
                assert i == len(args)-1
                # XXX we only want to localize the keys, not the values
                d = self._localize_keys(state, a.struct)
                if not isinstance(d, dict):
                    raise pql_TypeError(f"Expression to inline is not a map: {d}")
                for k, v in d.items():
                    inline_args.append(ast.NamedField(None, k, new_value_instance(v)))
            else:
                inline_args.append(ast.NamedField(None, None, a))

        args = inline_args
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
            matched.append((self.param_collector, MapInstance(collected)))
        return matched



@dataclass
class UserFunction(Function):
    name: str
    params: List[Param]
    expr: (ast.Expr, ast.CodeBlock)
    param_collector: Optional[Param]

    @property
    def docstring(self):
        if isinstance(self.expr, ast.CodeBlock):
            stmts = self.expr.statements
            if stmts and isinstance(stmts[0], ast.Const) and stmts[0].type is types.String:
                return stmts[0].value

@dataclass
class InternalFunction(Function):
    name: str
    params: List[Param]
    func: Callable
    param_collector: Optional[Param] = None

    meta = None     # Not defined in PQL code

    @property
    def docstring(self):
        return self.func.__doc__


# Instances

class AbsInstance(types.PqlObject):
    def get_attr(self, name):
        v = self.type.get_attr(name)
        if isinstance(v.type, types.FunctionType):
            # method
            return MethodInstance(self, v)
        else:
            return AttrInstance(self, v, name)

@dataclass
class MethodInstance(AbsInstance, Function):
    parent: AbsInstance
    func: Function

    params = property(X.func.params)
    expr = property(X.func.expr)

    # type = types.FunctionType()
    name = property(X.func.name)


@dataclass
class Instance(AbsInstance):
    code: sql.Sql
    type: types.PqlType

    subqueries: SafeDict

    @classmethod
    def make(cls, code, type_, instances, *extra):
        return cls(code, type_, merge_subqueries(instances), *extra)

    def repr(self, state):
        return f'<instance of {self.type.repr(state)}>'

    def __post_init__(self):
        assert not self.type.composed_of((types.StructType, types.Aggregated, types.TableType)), self

    def flatten_code(self):
        assert not self.type.composed_of(types.StructType)
        return [self.code]

    def primary_key(self):
        return self

def from_python(value):
    if value is None:
        return null
    elif isinstance(value, str):
        return ast.Const(None, types.String, value)
    elif isinstance(value, int):
        return ast.Const(None, types.Int, value)
    elif isinstance(value, list):
        return ast.List_(None, types.ListType(types.any_t), list(map(from_python, value)))
    elif isinstance(value, dict):
        return ast.Dict_(None, value)
    assert False, value


def new_value_instance(value, type_=None, force_type=False):
    r = sql.value(value)
    if force_type:
        assert type_
    elif type_:
        assert isinstance(type_, (types.Primitive, types.NullType, types.IdType)), type_
        assert r.type == type_, (r.type, type_)
    else:
        type_ = r.type
    if settings.optimize:   # XXX a little silly? But maybe good for tests
        return ValueInstance.make(r, type_, [], value)
    else:
        return Instance.make(r, type_, [])


@dataclass
class ValueInstance(Instance):
    local_value: object

    def get_attr(self, name):
        assert not isinstance(self.type, types.RowType)
            # try:
            #     obj = self.local_value[name]
            # except KeyError:
            #     raise pql_AttributeError(None, name)
            # return from_python(obj)   # XXX Maybe use 'code' to be more efficient?
        return super().get_attr(name)

    def repr(self, state):
        return self.type.repr_value(self.local_value)


@dataclass
class TableInstance(Instance):
    def __post_init__(self):
        assert isinstance(self.type, types.Collection), self.type

    @property
    def __columns(self):
        return {n:self.get_column(n) for n in self.type.columns}

    def get_column(self, name):
        # TODO memoize? columns shouldn't change
        t = self.type
        return make_instance_from_name(t.columns[name].col_type, t.column_codename(name))

    def all_attrs(self):
        # XXX hacky way to write it
        attrs = {n:self.get_attr(n) for n,f in self.type.dyn_attrs.items()}
        return SafeDict(attrs).update(self.__columns)


def make_instance_from_name(t, cn):
    if t.composed_of(types.StructType):
        return StructInstance(t, {n: make_instance_from_name(mt, types.join_names((cn, n))) for n,mt in t.members.items()})
    return make_instance(sql.Name(t, cn), t, [])

def make_instance(code, t, insts):
    assert not t.composed_of(types.StructType)
    if isinstance(t, types.Collection):
        return TableInstance.make(code, t, insts)
    elif isinstance(t, types.Aggregated):
        return AggregatedInstance(t, make_instance(code, t.elemtype, insts))
    else:
        return Instance.make(code, t, insts)


@dataclass
class AggregatedInstance(AbsInstance):
    type: types.PqlType
    elem: AbsInstance

    @property
    def code(self):
        return self.elem.code

    @property
    def subqueries(self):
        return self.elem.subqueries

    def get_attr(self, name):
        x = self.elem.get_attr(name)
        return make_instance(x.code, types.Aggregated(x.type), [x])

    def flatten_code(self):
        return self.elem.flatten_code()

    def primary_key(self):
        return self.elem.primary_key()


class AbsStructInstance(AbsInstance):
    def get_attr(self, name):
        return self.attrs[name]

    @property
    def code(self):
        # XXX this shouldn't even be allowed to happen in the first place
        raise pql_TypeError(None, "structs are abstract objects and cannot be sent to target. Choose one of its members instead.")


@dataclass
class StructInstance(AbsStructInstance):
    type: types.PqlType
    attrs: Dict[str, types.PqlObject]

    def __post_init__(self):
        assert self.type.composed_of((types.StructType, types.RowType)), self.type

    @property
    def subqueries(self):
        return merge_subqueries(self.attrs.values())

    def flatten_code(self):
        return [c for m in self.attrs.values() for c in m.flatten_code()]

    def primary_key(self):
        # XXX This is obviously wrong
        return list(self.attrs.values())[0]

    def all_attrs(self):
        # XXX hacky way to write it
        attrs = {n:self.get_attr(n) for n,f in self.type.dyn_attrs.items()}
        return SafeDict(attrs).update(self.attrs)

    def get_attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        else:
            raise pql_AttributeError(None, f"No such attribute: {name}")


@dataclass
class MapInstance(AbsStructInstance):
    attrs: Dict[str, types.PqlObject]

    type = types.any_t

    def __len__(self):
        return len(self.attrs)

    def items(self):
        return self.attrs.items()

    def primary_key(self):
        return self

    # def repr(self, state):
    #     from .evaluate import localize
    #     return repr(localize(state, self))
    def repr(self, state):
        inner = [f'{name}: {v.repr(state)}' for name, v in self.attrs.items()]
        return 'Map{%s}' % ', '.join(inner)


class RowInstance(StructInstance):
    def primary_key(self):
        return self.attrs['id']

        #     tbl_t = self.type.table
        #     col_t = tbl_t.columns[name].col_type
        #     code = sql.Select(col_t, sql.TableName(tbl_t, tbl_t.name), [sql.Name(col_t, name)], conds=[sql.Compare('=', [sql.Name(types.Int, 'id'), sql.value(self.attrs['id'])])])
        #     return make_instance(code, col_t, [])

    def repr(self, state):
        inner = [f'{name}: {v.repr(state)}' for name, v in self.attrs.items()]
        return 'Row{%s}' % ', '.join(inner)

@dataclass
class AttrInstance(AbsInstance):
    parent: AbsInstance
    type: types.PqlType
    name: str

    @property
    def subqueries(self):
        return self.parent.subqueries

    @property
    def code(self):
        raise pql_TypeError(None, f"Operation not supported for {self.repr(None)}")
    #     return self._resolve_attr().code

    def flatten_code(self):
        return self._resolve_attr().flatten_code()

    def get_attr(self, name):
        return self._resolve_attr().get_attr(name)

    def _resolve_attr(self):
        return self.parent.get_attr(self.name)

    def repr(self, state):
        p = self.parent.repr(state)
        return f'{p}.{self.name}'



def merge_subqueries(instances):
    return SafeDict().update(*[i.subqueries for i in instances])


def aggregated(inst):
    return AggregatedInstance(types.Aggregated(inst.type), inst)


null = ValueInstance.make(sql.null, types.null, [], None)

@dataclass
class EmptyListInstance(TableInstance):
    """Special case, because it is untyped
    """
    # def get_attr(self, name):
    #     return EmptyList

_empty_list_type = types.ListType(types.null)
# from collections import defaultdict
EmptyList = EmptyListInstance.make(sql.EmptyList(_empty_list_type), _empty_list_type, []) #, defaultdict(_any_column))    # Singleton


def aliased_table(t, name):
    assert isinstance(t, Instance)
    assert isinstance(t.type, types.Collection), t.type

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(sql.Name(t, n), types.join_names((name, n)))
        for (n, t) in t.type.flatten_type()
    ]

    code = sql.Select(t.type, t.code, sql_fields)
    return t.replace(code=code)


def new_table(type_, name=None, instances=None):
    return TableInstance.make(sql.TableName(type_, name or type_.name), type_, instances or [])