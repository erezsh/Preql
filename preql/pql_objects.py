"""
A collection of objects that may come to interaction with the user.
"""

from typing import List, Optional, Callable, Any, Dict

from .utils import dataclass, SafeDict, X, listgen, dy
from .exceptions import pql_AttributeError, Signal
from . import settings
from . import pql_ast as ast
from . import sql
from . import pql_types

from .pql_types import ITEM_NAME, T, Type, Object
from .types_impl import repr_value, flatten_type, join_names

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

    def repr(self, state):
        return f'<Module {self.name} | {len(self.namespace)} members>'

class Function(Object):

    @property
    def type(self):
        return T.function[tuple(p.type or T.any for p in self.params)](param_collector=self.param_collector is not None)

    def help_str(self, state):
        raise NotImplementedError()

    def repr(self, state):
        return '<%s>' % self.help_str(state)


    @listgen
    def match_params_fast(self, state, args):
        for i, p in enumerate(self.params):
            if i < len(args):
                v = args[i]
            else:
                v = p.default
                if v is None:
                    raise Signal.make(T.TypeError, state, None, f"Function '{self.name}' is missing a value for parameter '{p.name}'")


            yield p, v


    def _localize_keys(self, state, struct):
        raise NotImplementedError()

    def match_params(self, state, args):

        # If no keyword arguments, matching is much simpler and faster
        if all(not isinstance(a, (ast.NamedField, ast.InlineStruct)) for a in args):
            return self.match_params_fast(state, args)

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
                    raise Signal.make(T.TypeError, state, None, f"Expression to inline is not a map: {d}")
                for k, v in d.items():
                    inline_args.append(ast.NamedField(k, new_value_instance(v)))
            else:
                inline_args.append(ast.NamedField(None, a))

        args = inline_args
        named = [arg.name is not None for arg in args]
        try:
            first_named = named.index(True)
        except ValueError:
            first_named = len(args)
        else:
            if not all(n for n in named[first_named:]):
                # TODO meta
                raise Signal.make(T.TypeError, state, None, f"Function {self.name} received a non-named argument after a named one!")

        if first_named > len(self.params):
            # TODO meta
            raise Signal.make(T.TypeError, state, None, f"Function '{self.name}' takes {len(self.params)} parameters but received {first_named} arguments.")

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
                    raise Signal.make(T.TypeError, state, None, f"Function '{self.name}' has no parameter named '{arg_name}'")


        for name, value in values.items():
            if value is None:
                # TODO meta
                raise Signal.make(T.TypeError, state, None, f"Error calling function '{self.name}': parameter '{name}' has no value")

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
            if stmts and isinstance(stmts[0], ast.Const) and stmts[0].type is T.string:
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

class AbsInstance(Object):
    def get_attr(self, name):
        v = self.type.get_attr(name)
        if v <= T.function:
            return MethodInstance(self, v)

        raise pql_AttributeError(name)

@dataclass
class MethodInstance(AbsInstance, Function):
    parent: AbsInstance
    func: Function

    params = property(X.func.params)
    expr = property(X.func.expr)

    name = property(X.func.name)

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

    def repr(self, state):
        # Overwritten in evaluate.py
        raise NotImplementedError()
        #     return f'<instance of {self.type.repr(state)}>'

    def __post_init__(self):
        assert not self.type.issubtype(T.union[T.struct, T.aggregate, T.table, T.unknown])

    def flatten_code(self):
        assert not self.type.issubtype(T.struct)
        return [self.code]

    def primary_key(self):
        return self



def new_value_instance(value, type_=None, force_type=False):
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
    else:
        return Instance.make(r, type_, [])


@dataclass
class ValueInstance(Instance):
    local_value: object

    def repr(self, state):
        return repr_value(state, self)

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
        return {n:self.get_column(n) for n in self.type.elems}

    def get_column(self, name):
        # TODO memoize? columns shouldn't change
        t = self.type.elems
        return make_instance_from_name(t[name], name) #t.column_codename(name))

    def all_attrs(self):
        # XXX hacky way to write it
        attrs = dict(self.type.methods)
        return SafeDict(attrs).update(self.__columns)

    def get_attr(self, name):
        try:
            v = self.type.elems[name]
            return SelectedColumnInstance(self, v, name)
        except KeyError:
            try:
                return MethodInstance(self, self.type.methods[name])
            except KeyError:
                raise pql_AttributeError(name)

@dataclass
class ListInstance(CollectionInstance):
    def __post_init__(self):
        assert self.type <= T.list, self.type

    def get_column(self, name):
        # TODO memoize? columns shouldn't change
        assert name == ITEM_NAME
        t = self.type
        return make_instance_from_name(t.elem, name)

    def all_attrs(self):
        # XXX hacky way to write it
        attrs = dict(self.type.methods)
        attrs[ITEM_NAME] = self.get_column(ITEM_NAME)
        return attrs

    def get_attr(self, name):
        if name == ITEM_NAME:
            v = self.type.elem
            return SelectedColumnInstance(self, v, name)
        else:
            try:
                return MethodInstance(self, self.type.methods[name])
            except KeyError:
                raise pql_AttributeError(name)



def make_instance_from_name(t, cn):
    if t <= T.struct:
        return StructInstance(t, {n: make_instance_from_name(mt, join_names((cn, n))) for n,mt in t.elems.items()})
    return make_instance(sql.Name(t, cn), t, [])

def make_instance(code, t, insts):
    assert not t.issubtype(T.struct), t
    if t <= T.list:
        return ListInstance.make(code, t, insts)
    elif t <= T.table:
        return TableInstance.make(code, t, insts)
    elif t <= T.aggregate:
        return AggregateInstance(t, make_instance(code, t.elem, insts))
    elif t <= T.unknown:
        return unknown
    else:
        return Instance.make(code, t, insts)


@dataclass
class AggregateInstance(AbsInstance):
    type: Type
    elem: AbsInstance

    @property
    def code(self):
        return self.elem.code

    @property
    def subqueries(self):
        return self.elem.subqueries

    def get_attr(self, name):
        x = self.elem.get_attr(name)
        return make_instance(x.code, T.aggregate[x.type], [x])

    def all_attrs(self):
        return self.elem.all_attrs()

    def flatten_code(self):
        return self.elem.flatten_code()

    def primary_key(self):
        # TODO should return aggregate key, no?
        return self.elem.primary_key()


class AbsStructInstance(AbsInstance):
    def get_attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        else:
            raise pql_AttributeError(name)

    @property
    def code(self):
        # XXX this shouldn't even be allowed to happen in the first place
        raise Signal(T.TypeError, [], "structs are abstract objects and cannot be sent to target. Choose one of its members instead.")


@dataclass
class StructInstance(AbsStructInstance):
    type: Type
    attrs: Dict[str, Object]

    def __post_init__(self):
        assert self.type <= T.union[T.struct, T.vectorized[T.struct]]

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

    def repr(self, state):
        attrs = [f'{k}: {v.repr(state)}' for k, v in self.attrs.items()]
        return '{%s}' % ', '.join(attrs)



@dataclass
class MapInstance(AbsStructInstance):
    attrs: Dict[str, Object]

    type = T.struct

    def __len__(self):
        return len(self.attrs)

    def items(self):
        return self.attrs.items()

    def all_attrs(self):
        return dict(self.attrs)

    def primary_key(self):
        return self

    def repr(self, state):
        inner = [f'{name}: {v.repr(state)}' for name, v in self.attrs.items()]
        return 'Map{%s}' % ', '.join(inner)


class RowInstance(StructInstance):
    def primary_key(self):
        try:
            return self.attrs['id']
        except KeyError:
            # XXX this is a hack!
            return list(self.attrs.values())[0]

    def repr(self, state):
        inner = [f'{name}: {v.repr(state)}' for name, v in self.attrs.items()]
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

    def replace(self, **kw):
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
        raise Signal(T.TypeError, [], f"Operation not supported for {self}")
    #     return self._resolve_attr().code

    def flatten_code(self):
        return self._resolve_attr().flatten_code()

    def get_attr(self, name):
        return self._resolve_attr().get_attr(name)

    def _resolve_attr(self):
        return self.parent.get_column(self.name)

    def repr(self, state):
        p = self.parent.repr(state)
        return f'{p}.{self.name}'



def merge_subqueries(instances):
    return SafeDict().update(*[i.subqueries for i in instances])


def aggregate(inst):
    if isinstance(inst, AbsInstance):
        return AggregateInstance(T.aggregate[inst.type], inst)

    return inst

def vectorized(inst):
    if not isinstance(inst, AbsInstance):
        return inst
    if inst.type <= T.vectorized:
        return inst
    return inst.replace(type=T.vectorized[inst.type])

def unvectorized(inst):
    assert inst.type <= T.vectorized
    return inst.replace(type=inst.type.elem)


@dy
def unvectorize_args(x: list):
    if not x:
        return False, x
    was_vec, objs = zip(*[unvectorize_args(i) for i in x])
    return any(was_vec), list(objs)

@dy
def unvectorize_args(x: AbsInstance):
    if x.type <= T.vectorized:
        return True, unvectorized(x)
    return False, x

@dy
def unvectorize_args(x):
    return False, x


null = ValueInstance.make(sql.null, T.nulltype, [], None)

@dataclass
class EmptyListInstance(ListInstance):
    """Special case, because it is untyped
    """

_empty_list_type = T.list[T.nulltype]
EmptyList = EmptyListInstance.make(sql.EmptyList(_empty_list_type), _empty_list_type, []) #, defaultdict(_any_column))    # Singleton


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
    if type_ <= T.list:
        cls = ListInstance
    else:
        cls = TableInstance
    name = name if name else type_.options.get('name', sql.Id('anon'))
    inst = cls.make(sql.TableName(type_, name), type_, instances or [])

    if select_fields:
        code = sql.Select(type_, inst.code, [sql.Name(t, n) for n, t in type_.elems.items()])
        inst = inst.replace(code=code)

    return inst

def new_const_table(state, table_type, tuples):
    name = state.unique_name("table_")
    table_code, subq = sql.create_table(table_type, name, tuples)

    inst = TableInstance.make(table_code, table_type, [])
    inst.subqueries[name] = subq
    return inst



class PythonList(ast.Ast):
    # TODO just a regular const?
    def __init__(self, items):
        types = set(type(i) for i in items)
        # TODO if not one type, raise typeerror
        type_ ,= types
        self.type = T.list[pql_types.from_python(type_)]

        # allow to compile it straight to SQL, no AST in the middle
        self.items = items

def from_python(value):
    if value is None:
        return null
    elif isinstance(value, str):
        return ast.Const(T.string, value)
    elif isinstance(value, bool):
        return ast.Const(T.bool, value)
    elif isinstance(value, int):
        return ast.Const(T.int, value)
    elif isinstance(value, list):
        # return ast.List_(T.list[T.any], list(map(from_python, value)))
        return PythonList(value)
    elif isinstance(value, dict):
        #return ast.Dict_(value)
        elems = {k:from_python(v) for k,v in value.items()}
        return ast.Dict_(elems)
    assert False, value

