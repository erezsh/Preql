import rich

from runtype import dataclass

from preql.utils import safezip

from preql.docstring.docstring import parse, Section, Defin
from preql import Preql, T


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

    def print_text(self, indent=0):
        params = [str(p.name) for p in self.func.params]
        if self.func.param_collector:
            params.append('...' + self.func.param_collector.name)
        params = ', '.join(params)
        indent_str = ' ' * indent
        s = f'{indent_str}[dodger_blue2]func[/dodger_blue2] [bold white]{self.func.name}[/bold white]({params}) = ...\n\n'
        return s + self.doc.print_text(indent+4)

    def print_rst(self):
        params = [str(p.name) for p in self.func.params]
        if self.func.param_collector:
            params.append('...' + self.func.param_collector.name)
        params = ', '.join(params)
        s = f".. function:: {self.func.name}({params})\n\n"
        return s + self.doc.print_rst()


from lark import LarkError

def doc_func(f):
    assert f.docstring

    try:
        doc_tree = parse(f.docstring)
    except LarkError as e:
        raise ValueError(f"Error in docstring of function {f.name}")


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
            d.default = p.default.repr(None) if p.default else ''

    return FuncDoc(f, doc_tree)


def doc_module(m):
    funcs = [v for v in m.namespace.values() if v.type <= T.function and v.docstring and not v.name.startswith('_')]
    funcs.sort(key=lambda f: f.name)
    return ModuleDoc(m, list(map(doc_func, funcs)))


def test_func():
    p = Preql()
    rich.print(doc_func(p('bfs')).print_text())

def test_module():
    p = Preql()
    rich.print(doc_module(p('__builtins__')).print_text())

def generate_rst(filename):
    p = Preql()
    with open(filename, 'w', encoding='utf8') as f:
        print('Preql Modules', file=f)
        print('=============', file=f)
        print(doc_module(p('__builtins__')).print_rst(), file=f)

# test_func()
# test_module()
if __name__ == '__main__':
    generate_rst('preql-api.rst')