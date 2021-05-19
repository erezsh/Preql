"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable, Any, Dict

from preql.utils import dataclass, SafeDict, X, listgen
from preql import settings

from .exceptions import pql_AttributeError, Signal
from . import pql_ast as ast
from . import sql
from . import pql_types
from .state import unique_name

from .pql_types import T, Type, Object, dp_inst
from .types_impl import flatten_type, join_names, pql_repr


# Functions
@dataclass
class Param(ast.Ast):
    name: str
    type: Optional[Object] = None
    default: Optional[Object] = None
    orig: Any = None # XXX temporary and lazy, for TableConstructor

class ParamVariadic(Param):
    pass


@dataclass
class ParamDict(Object):
    params: Dict[str, Object]

    def __len__(self):
        return len(self.params)

    def items(self):
        return self.params.items()

    @property
    def type(self):
        return tuple((n,p.type) for n,p in self.params.items())

@dataclass
class Module(Object):
    name: str
    namespace: dict

    def get_attr(self, attr):
        try:
            return self.namespace[attr]
        except KeyError:
            raise pql_AttributeError(attr)

    def all_attrs(self):
        return self.namespace

    @property
    def type(self):
        return T.module

    def __repr__(self):
        return f'<preql:Module | {len(self.namespace)} members>'

    def public_functions(self):
        funcs = [v for v in self.namespace.values()
                 if v.type <= T.function and v.docstring and not v.name.startswith('_')]
        funcs.sort(key=lambda f: f.name)
        return funcs


class Function(Object):
    # Abstract class
    name: str
    params: List[Param]
    expr: (ast.Expr, ast.CodeBlock)
    param_collector: Optional[Param]
    docstring: Optional[str]

    @property
    def type(self):
        return T.function[tuple(p.type or T.any for p in self.params)](param_collector=self.param_collector is not None)

    def help_str(self):
        params = []
        for p in self.params:
            s = p.name
            if p.type:
                s += f": {p.type}"
            if p.default:
                s += f"={p.default.repr()}"
            params.append(s)

        if self.param_collector is not None:
            params.append(f"...{self.param_collector.name}")
        param_str = ', '.join(params)
        return f"func {self.name}({param_str}) = ..."

    def __repr__(self):
        return f'<preql:Function | {self.name}: {self.type}>'

    @listgen
    def match_params_fast(self, args):
        for i, p in enumerate(self.params):
            if i < len(args):
                v = args[i]
            else:
                v = p.default
                if v is None:
                    msg = f"Function '{self.name}' is missing a value for parameter '{p.name}'"
                    raise Signal.make(T.TypeError, None, msg)


            yield p, v

        if self.param_collector:
            yield self.param_collector, ast.Dict_({})


    def _localize_keys(self, struct):
        raise NotImplementedError()

    def match_params(self, args):
        # if not self.name.startswith('_') and self.name not in ('PY', 'SQL', 'get_db_type'):
        #     breakpoint()

        # If no keyword arguments, matching is much simpler and faster
        if all(not isinstance(a, (ast.NamedField, ast.Ellipsis)) for a in args):
            return self.match_params_fast(args)

        # Canonize args for the rest of the function
        inline_args = []
        for i, a in enumerate(args):
            if isinstance(a, ast.NamedField):
                inline_args.append(a)
            elif isinstance(a, ast.Ellipsis):
                assert i == len(args)-1
                if a.exclude:
                    raise NotImplementedError("Cannot exclude keys when inlining struct")

                # XXX we only want to localize the keys, not the values
                # TODO remove this?
                d = self._localize_keys(a.from_struct)
                if not isinstance(d, dict):
                    raise Signal.make(T.TypeError, None, f"Expression to inline is not a map: {d}")
                for k, v in d.items():
                    inline_args.append(ast.NamedField(k, pyvalue_inst(v)))
            else:
                inline_args.append(ast.NamedField(None, a, user_defined=False))

        args = inline_args
        named = [arg.name is not None for arg in args]
        try:
            first_named = named.index(True)
        except ValueError:
            first_named = len(args)
        else:
            if not all(n for n in named[first_named:]):
                # TODO meta
                msg = f"Function {self.name} received a non-named argument after a named one!"
                raise Signal.make(T.TypeError, None, msg)

        if first_named > len(self.params):
            # TODO meta
            msg = f"Function '{self.name}' takes {len(self.params)} parameters but received {first_named} arguments."
            raise Signal.make(T.TypeError, None, msg)

        values = {p.name: p.default for p in self.params}

        names_set = set()
        for pos_arg, name in zip(args[:first_named], values):
            assert pos_arg.name is None
            values[name] = pos_arg.value
            names_set.add(name)

        collected = {}
        if first_named is not None:
            for named_arg in args[first_named:]:
                arg_name = named_arg.name
                if arg_name in values:
                    if arg_name in names_set:
                        raise Signal.make(T.SyntaxError, None, f"Function '{self.name}' recieved argument '{arg_name}' both as keyword and as positional.")

                    names_set.add(arg_name)
                    values[arg_name] = named_arg.value
                elif self.param_collector:
                    assert arg_name not in collected
                    collected[arg_name] = named_arg.value
                else:
                    # TODO meta
                    raise Signal.make(T.TypeError, None, f"Function '{self.name}' has no parameter named '{arg_name}'")


        for name, value in values.items():
            if value is None:
                # TODO meta
                msg = f"Error calling function '{self.name}': parameter '{name}' has no value"
                raise Signal.make(T.TypeError, None, msg)

        matched = [(p, values.pop(p.name)) for p in self.params]
        assert not values, values
        if self.param_collector:
            matched.append((self.param_collector, ast.Dict_(collected)))
        return matched



@dataclass
class UserFunction(Function):
    name: str
    params: List[Param]
    expr: (ast.Expr, ast.CodeBlock)
    param_collector: Optional[Param]
    docstring: Optional[str]


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

@dataclass
class Property(Object):
    func: Function

    type = T.property
    name = property(X.func.name)


# post_instance_getattr. Property handling is specified in evaluate

@dp_inst
def post_instance_getattr(inst, obj):
    return obj

@dp_inst
def post_instance_getattr(inst, f: T.function):
    return MethodInstance(inst, f)



# Instances

class AbsInstance(Object):
    def get_attr(self, name):
        v = self.type.get_attr(name)
        return post_instance_getattr(self, v)

@dataclass
class MethodInstance(AbsInstance, Function):
    parent: AbsInstance
    func: Function

    params = property(X.func.params)
    expr = property(X.func.expr)
    param_collector = property(X.func.param_collector)

    name = property(X.func.name)

@dataclass
class PropertyInstance(AbsInstance):
    parent: AbsInstance
    func: Function

    name = property(X.func.name)
    type = T.any

@dataclass
class ExceptionInstance(AbsInstance):
    exc: Exception



@dataclass
class Instance(AbsInstance):
    code: sql.Sql
    type: Type

    subqueries: SafeDict

    @classmethod
    def make(cls, code, type_, instances, *extra):
        return cls(code, type_, merge_subqueries(instances), *extra)

    def repr(self):
        # Overwritten in evaluate.py
        raise NotImplementedError()
        #     return f'<instance of {self.type.repr(state)}>'

    def __post_init__(self):
        assert not self.type.issubtype(T.union[T.struct, T.table, T.unknown])

    def flatten_code(self):
        assert not self.type.issubtype(T.struct)
        return [self.code]

    def primary_key(self):
        return self



def pyvalue_inst(value, type_=None, force_type=False):

    r = sql.make_value(value)

    if force_type:
        assert type_
    elif type_:
        assert type_ <= T.union[T.primitive, T.nulltype, T.t_id]
        assert r.type == type_, (r.type, type_)
    else:
        type_ = r.type

    if settings.optimize:   # XXX a little silly? But maybe good for tests
        return ValueInstance.make(r, type_, [], value)
    return Instance.make(r, type_, [])


@dataclass
class ValueInstance(Instance):
    local_value: object

    def repr(self):
        return pql_repr(self.type, self.local_value)

    @property
    def value(self):
        return self.local_value


class CollectionInstance(Instance):
    pass

@dataclass
class TableInstance(CollectionInstance):
    def __post_init__(self):
        assert self.type <= T.table, self.type #and not self.type <= T.list, self.type

    @property
    def __columns(self):
        return {n:self.get_column(n) for n in self.type.elems.keys()}

    def get_column(self, name):
        # TODO memoize? columns shouldn't change
        t = self.type.elems
        return make_instance_from_name(t[name], name) #t.column_codename(name))

    def all_attrs(self):
        attrs = SafeDict(self.type.proto_attrs)
        return attrs.update(self.__columns)

    def get_attr(self, name):
        try:
            v = self.type.elems[name]
            return SelectedColumnInstance(self, v, name)
        except KeyError:
            return super().get_attr(name)




def make_instance_from_name(t, cn):
    if t <= T.struct:
        return StructInstance(t, {n: make_instance_from_name(mt, join_names((cn, n))) for n,mt in t.elems.items()})
    return make_instance(sql.Name(t, cn), t, [])

def make_instance(code, t, insts):
    if t.issubtype(T.struct):
        raise Signal.make(T.TypeError, t, "Cannot instanciate structs directly")

    assert not t.issubtype(T.struct), t
    if t <= T.table:
        return TableInstance.make(code, t, insts)
    elif t <= T.unknown:
        return unknown

    return Instance.make(code, t, insts)



class AbsStructInstance(AbsInstance):
    type: Type
    attrs: Dict[str, Object]

    def get_attr(self, name):
        if name in self.attrs:
            attr = self.attrs[name]
            return inherit_phantom_type(attr, [self])

        raise pql_AttributeError(name)

    @property
    def code(self):
        # XXX this shouldn't even be allowed to happen in the first place
        msg = "structs are abstract objects and cannot be sent to target. Choose one of its members instead."
        raise Signal.make(T.TypeError, None, msg)


@dataclass
class StructInstance(AbsStructInstance):
    type: Type
    attrs: Dict[str, Object]

    def __post_init__(self):
        assert self.type <= T.struct

    @property
    def subqueries(self):
        return merge_subqueries(self.attrs.values())

    def flatten_code(self):
        return [c for m in self.attrs.values() for c in m.flatten_code()]

    def primary_key(self):
        # XXX This is obviously wrong
        return list(self.attrs.values())[0]

    def all_attrs(self):
        return self.attrs

    def repr(self):
        attrs = [f'{k}: {v.inline_repr()}' for k, v in self.attrs.items()]
        return '{%s}' % ', '.join(attrs)

    def values(self):
        return self.attrs.values()

    def __iter__(self):
        return iter(self.attrs)


# TODO simplify with mixin
# @dataclass
# class MapInstance(AbsStructInstance):
#     attrs: Dict[str, Object]

#     type = T.struct

#     def __len__(self):
#         return len(self.attrs)

#     def items(self):
#         return self.attrs.items()

#     def __iter__(self):
#         return iter(self.attrs)

#     def keys(self):
#         return self.attrs.keys()

#     def values(self):
#         return self.attrs.values()

#     def all_attrs(self):
#         return dict(self.attrs)

#     def primary_key(self):
#         return self

#     def repr(self):
#         inner = [f'{name}: {v.repr()}' for name, v in self.attrs.items()]
#         return 'Map{%s}' % ', '.join(inner)


class RowInstance(StructInstance):
    def primary_key(self):
        try:
            return self.attrs['id']
        except KeyError:
            # XXX this is a hack!
            return list(self.attrs.values())[0]

    def repr(self):
        inner = [f'{name}: {v.repr()}' for name, v in self.attrs.items()]
        return 'Row{%s}' % ', '.join(inner)


class UnknownInstance(AbsInstance):
    type = T.unknown
    subqueries = {}
    code = sql.unknown

    def get_attr(self, name):
        return self # XXX use name?

    def all_attrs(self):
        return {}

    def flatten_code(self):
        return [self.code]

    def replace(self, **_kw):
        # XXX Is this right?
        return self


unknown = UnknownInstance()

@dataclass
class SelectedColumnInstance(AbsInstance):
    parent: CollectionInstance
    type: Type
    name: str

    @property
    def subqueries(self):
        return self.parent.subqueries

    @property
    def code(self):
        raise Signal.make(T.TypeError, [], f"Operation not supported for {self}")
    #     return self._resolve_attr().code

    def flatten_code(self):
        return self._resolve_attr().flatten_code()

    def get_attr(self, name):
        return self._resolve_attr().get_attr(name)

    def _resolve_attr(self):
        return self.parent.get_column(self.name)

    def repr(self):
        try:
            p = self.parent.type.options['name'].repr_name
        except KeyError:
            p = '?'
        return f'{p}.{self.name}'



def merge_subqueries(instances):
    return SafeDict().update(*[i.subqueries for i in instances])


def ensure_phantom_type(inst, ptype):
    if not isinstance(inst, AbsInstance):
        return inst
    if inst.type <= ptype:
        return inst
    return inst.replace(type=ptype[inst.type])

def aggregate(inst):
    return ensure_phantom_type(inst, T.aggregated)

def projected(inst):
    return ensure_phantom_type(inst, T.projected)


def remove_phantom_type(inst):
    if inst.type <= T.projected | T.aggregated:
        return inst.replace(type=inst.type.elem)
    return inst

def inherit_vectorized_type(t, objs):
    # XXX reevaluate this function
    for src in objs:
        if src.type <= T.projected:
            return T.projected[t]
    return t

def inherit_phantom_type(o, objs):
    for src in objs:
        if src.type <= T.projected | T.aggregated:
            return ensure_phantom_type(o, src.type)
    return o



null = ValueInstance.make(sql.null, T.nulltype, [], None)

@dataclass
class EmptyListInstance(TableInstance):
    """Special case, because it is untyped
    """

_empty_list_type = T.list[T.nulltype]
EmptyList = EmptyListInstance.make(sql.EmptyList(_empty_list_type), _empty_list_type, [])


def alias_table_columns(t, prefix):
    assert isinstance(t, CollectionInstance)
    assert t.type <= T.table

    # Make code
    sql_fields = [
        sql.ColumnAlias.make(sql.Name(t, n), join_names((prefix, n)))
        for (n, t) in flatten_type(t.type)
    ]

    code = sql.Select(t.type, t.code, sql_fields)
    return t.replace(code=code)


def new_table(type_, name=None, instances=None, select_fields=False):
    "Create new table instance"
    name = name if name else type_.options.get('name', sql.Id('anon'))
    inst = TableInstance.make(sql.TableName(type_, name), type_, instances or [])

    if select_fields:
        code = sql.Select(type_, inst.code, [sql.Name(t, n) for n, t in type_.elems.items()])
        inst = inst.replace(code=code)

    return inst

def new_const_table(table_type, tuples):
    name = unique_name("table_")
    table_code, subq = sql.create_table(table_type, name, tuples)

    inst = TableInstance.make(table_code, table_type, [])
    inst.subqueries[name] = subq
    return inst



class PythonList(ast.Ast):
    # TODO just a regular const?
    def __init__(self, items):
        types = set(type(i) for i in items)
        if not types:
            assert not items
            # self.type = T.list[T.any]
            # self.items = []
            # return
            raise NotImplementedError("Cannot import an empty list (no type deduced)")

        if len(types) > 1:
            raise ValueError("Expecting all items of the list to be of the same type")
        # TODO if not one type, raise typeerror
        type_ ,= types
        self.type = T.list[pql_types.from_python(type_)]

        # allow to compile it straight to SQL, no AST in the middle
        self.items = items


from preql.utils import dsp


@dsp
def from_python(value: type(None)):
    assert value is None
    return null

@dsp
def from_python(value: str):
    return ast.Const(T.string, value)

@dsp
def from_python(value: bytes):
    return ast.Const(T.string, value.decode())

@dsp
def from_python(value: bool):
    return ast.Const(T.bool, value)

@dsp
def from_python(value: int):
    return ast.Const(T.int, value)

@dsp
def from_python(value: float):
    return ast.Const(T.float, value)

@dsp
def from_python(value: list):
    return PythonList(value)

@dsp
def from_python(value: dict):
    elems = {k:from_python(v) for k,v in value.items()}
    return ast.Dict_(elems)

@dsp
def from_python(value: type):
    return pql_types.from_python(value)

@dsp
def from_python(value: Object):
        return value

@dsp
def from_python(value):
    raise Signal.make(T.TypeError, None, f"Cannot import into Preql a Python object of type {type(value)}")
