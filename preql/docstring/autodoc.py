from typing import Optional

from runtype import dataclass

from preql.utils import safezip, dsp

from preql.docstring.docstring import parse, Section, Defin, Text

from preql.core.pql_objects import Module, Function, T, MethodInstance
from preql.core.pql_types import Type, subtypes

from . import type_docs


class AutoDocError(Exception):
    pass

@dataclass
class ModuleDoc:
    module: object
    items: list

    def print_text(self):
        s = f'\n[dodger_blue2]module[/dodger_blue2] [bold white]{self.module.name}[/bold white]\n'
        line = '=' * (len(self.module.name) + 7)
        s += f'[dodger_blue2]{line}[/dodger_blue2]\n\n\n'
        return s + '\n\n'.join(i.print_text(2) for i in self.items)

    def print_rst(self):
        s = f'\n{self.module.name}\n'
        line = '-' * len(self.module.name)
        s += f'{line}\n\n\n'
        items = [i.print_rst() for i in self.items]
        return s + '\n\n'.join(items)


@dataclass
class FuncDoc:
    func: object
    doc: object
    parent_type: Optional[Type]

    def _print_text(self, indent):
        return self.print_text(indent)

    def _print_rst(self):
        return self.print_rst()

    def print_text(self, indent=0):
        params = [str(p.name) for p in self.func.params]
        if self.func.param_collector:
            params.append('...' + self.func.param_collector.name)
        params = ', '.join(params)
        indent_str = ' ' * indent
        parent = (self.parent_type.repr() + '.') if self.parent_type else ''
        s = f'{indent_str}[dodger_blue2]func[/dodger_blue2] {parent}[bold white]{self.func.name}[/bold white]({params}) = ...\n\n'
        return s + self.doc.print_text(indent+4)

    def print_rst(self):
        is_method = bool(self.parent_type)

        params = [str(p.name) for p in self.func.params]
        if self.func.param_collector:
            params.append('...' + self.func.param_collector.name)
        params = ', '.join(params)
        # parent = (self.parent_type.repr() + '.') if is_method else ''
        func_or_method = 'method' if is_method else 'function'
        s = f".. {func_or_method}:: {self.func.name}({params})\n\n"
        s = s + self.doc.print_rst()
        if is_method:
            s = '  ' + s.replace('\n', '\n  ')  # XXX hack to indent methods
        return s

@dataclass
class TypeDoc:
    type: object
    doc: object

    def print_text(self, indent=0):
        params = [str(p) for p in self.type.elems]
        params = ', '.join(params)
        if params:
            params = f'\\[{params}]'
        indent_str = ' ' * indent
        s = f'{indent_str}[dodger_blue2]type[/dodger_blue2] [bold white]{self.type.typename}[/bold white]{params}\n\n'
        return s + self.doc.print_text(indent+4)

    def print_rst(self):
        type_name = str(self.type)
        # s = type_name + '\n'
        # s += '^' * len(type_name) + '\n'

        s = f".. class:: {type_name}⁣\n\n"     # includes an invisible unicode separator to trick sphinx
        return s + self.doc.print_rst()



from lark import LarkError

def doc_func(f, parent_type=None):
    if isinstance(f, MethodInstance):
        f = f.func
    try:
        doc_tree = parse(f.docstring or '')
    except LarkError as e:
        raise AutoDocError(f"Error in docstring of function {f.name}: {e}")

    assert {s.name for s in doc_tree.sections} <= {'Parameters', 'Example', 'Examples', 'Note', 'Returns', 'See Also'}, [s.name for s in doc_tree.sections]
    try:
        params_doc = doc_tree.get_section('Parameters')
    except KeyError:
        if f.params:
            params_doc = Section('Parameters', [Defin(p.name, None, str(p.type) if p.type else '') for p in f.params])
            doc_tree.sections.insert(0, params_doc)
    else:
        params = list(f.params)
        if f.param_collector:
            params.append(f.param_collector)
        if len(params) != len(params_doc.items):
            raise AutoDocError(f"Parameters don't match docstring in function {f}")

        for d, p in safezip(params_doc.items, params):
            assert d.name == p.name, (d.name, p.name)
            d.type = str(p.type) if p.type else ''
            d.default = p.default.repr() if p.default else ''

    return FuncDoc(f, doc_tree, parent_type=parent_type)


def doc_module(m):
    return ModuleDoc(m, list(map(doc_func, m.public_functions())))


# def test_func():
#     p = Preql()
#     rich.print(doc_func(p('bfs')).print_text())

# def test_module():
#     p = Preql()
#     rich.print(doc_module(p('__builtins__')).print_text())

def generate_rst(modules_fn, types_fn):
    from preql import Preql
    p = Preql()

    with open(types_fn, 'w', encoding='utf8') as f:
        print('Preql Types', file=f)
        print('===========', file=f)
        for t in T.values():
            try:
                print(autodoc(t).print_rst(), file=f)
            except NotImplementedError:
                pass

    with open(modules_fn, 'w', encoding='utf8') as f:
        print('Preql Modules', file=f)
        print('=============', file=f)
        print(doc_module(p('__builtins__')).print_rst(), file=f)
        p('import graph')
        print(doc_module(p('graph')).print_rst(), file=f)

@dsp
def autodoc(m: Module):
    return doc_module(m)

@dsp
def autodoc(f: Function):
    return doc_func(f)

@dsp
def autodoc(t: Type):
    try:
        docstr = type_docs.DOCS[t]
    except KeyError:
        raise NotImplementedError(t)
    try:
        doc_tree = parse(docstr)
    except LarkError as e:
        raise AutoDocError(f"Error in docstring of type {t}")

    assert {s.name for s in doc_tree.sections} <= {'Example', 'Examples', 'Note', 'See Also'}, [s.name for s in doc_tree.sections]

    if t.proto_attrs:
        methods_doc = Section('Methods', [doc_func(f, t) for f in t.proto_attrs.values() if isinstance(f, Function)])
        doc_tree.sections.insert(0, methods_doc)

    if t in subtypes:
        subtypes_doc = Section('Subtypes', [Text([str(st) + ", "]) for st in subtypes[t]])
        doc_tree.sections.insert(0, subtypes_doc)


    if t.supertypes:
        supertypes_doc = Section('Supertypes', [Text([str(st)]) for st in t.supertypes])
        doc_tree.sections.insert(0, supertypes_doc)


    return TypeDoc(t, doc_tree)

# test_func()
# test_module()
if __name__ == '__main__':
    generate_rst('preql-modules.rst', 'preql-types.rst')
