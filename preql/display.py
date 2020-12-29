import html

import rich.table
import rich.text
import rich.console

from .exceptions import Signal
from .pql_types import T, ITEM_NAME
from . import pql_objects as objects
from . import pql_ast as ast
from .types_impl import dp_type, pql_repr
from .interp_common import call_pql_func, cast_to_python

TABLE_PREVIEW_SIZE = 16
LIST_PREVIEW_SIZE = 128
MAX_AUTO_COUNT = 10000

@dp_type
def pql_repr(state, t: T.function, value):
    params = []
    for p in value.params:
        s = p.name
        if p.type:
            s += ": %s" % p.type
        params.append(s)

    return f'{{func {value.name}({", ".join(params)})}}'

@dp_type
def pql_repr(state, t: T.decimal, value):
    raise Signal.make(T.NotImplementedError, state, None, "Decimal not implemented")

@dp_type
def pql_repr(state, t: T.string, value):
    assert isinstance(value, str), value
    value = value.replace('"', r'\"')
    res = f'"{value}"'
    if state.fmt == 'html':
        res = html.escape(res)
    return res

@dp_type
def pql_repr(state, t: T.text, value):
    assert isinstance(value, str), value
    return str(value)

@dp_type
def pql_repr(state, t: T.bool, value):
    return 'true' if value else 'false'

@dp_type
def pql_repr(state, t: T.nulltype, value):
    return 'null'




def table_limit(table, state, limit, offset=0):
    return call_pql_func(state, 'limit_offset', [table, ast.make_const(limit), ast.make_const(offset)])

def _call_pql_func(state, name, args):
    count = call_pql_func(state, name, args)
    return cast_to_python(state, count)

def _html_table(name, count_str, rows, offset):
    header = 'table '
    if name:
        header += name
    if offset:
        header += f'[{offset}..]'
    header += f" {count_str}"
    header = f"<pre>table {name}, {count_str}</pre>"

    if not rows:
        return header

    cols = list(rows[0])
    ths = '<tr>%s</tr>' % ' '.join([f"<th>{col}</th>" for col in cols])
    trs = [
        '<tr>%s</tr>' % ' '.join([f"<td>{v}</td>" for v in row.values()])
        for row in rows
    ]

    return '%s<table>%s%s</table>' % (header, ths, '\n'.join(trs))


def _rich_table(name, count_str, rows, offset, has_more, colors=True, show_footer=False):
    header = 'table '
    if name:
        header += name
    if offset:
        header += f'[{offset}..]'
    header += f" {count_str}"

    if not rows:
        return header

    table = rich.table.Table(title=rich.text.Text(header), show_footer=show_footer, min_width=len(header))

    # TODO enable/disable styling
    for k, v in rows[0].items():
        kw = {}
        if isinstance(v, (int, float)):
            kw['justify']='right'

        if colors:
            if isinstance(v, int):
                kw['style']='cyan'
            elif isinstance(v, float):
                kw['style']='yellow'
            elif isinstance(v, str):
                kw['style']='green'

        table.add_column(k, footer=k, **kw)

    for r in rows:
        table.add_row(*[rich.markup.escape(str(x) if x is not None else '-') for x in r.values()])

    if has_more:
        table.add_row(*['...' for x in rows[0]])

    return table

_g_last_table = None
_g_last_offset = 0

def table_repr(self, state, offset=0):
    global _g_last_table, _g_last_offset

    count = _call_pql_func(state, 'count', [table_limit(self, state, MAX_AUTO_COUNT)])
    if count == MAX_AUTO_COUNT:
        count_str = f'>={count}'
    else:
        count_str = f'={count}'

    # if len(self.type.elems) == 1:
    #     rows = cast_to_python(state, table_limit(self, state, LIST_PREVIEW_SIZE))
    #     post = f', ... ({count_str})' if len(rows) < count else ''
    #     elems = ', '.join(repr_value(ast.Const(None, self.type.elem, r)) for r in rows)
    #     return f'[{elems}{post}]'

    # TODO load into preql and repr, instead of casting to python
    rows = cast_to_python(state, table_limit(self, state, TABLE_PREVIEW_SIZE, offset))
    _g_last_table = self
    _g_last_offset = offset + len(rows)
    if self.type <= T.list:
        rows = [{ITEM_NAME: x} for x in rows]

    has_more = offset + len(rows) < count

    try:
        table_name = self.type.options['name'].repr_name
    except KeyError:
        table_name = ''

    if state.fmt == 'html':
        return _html_table(table_name, count_str, rows, offset)
    elif state.fmt == 'rich':
        return _rich_table(table_name, count_str, rows, offset, has_more)

    assert state.fmt == 'text'
    return _rich_table(table_name, count_str, rows, offset, has_more, colors=False)

    # raise NotImplementedError(f"Unknown format: {state.fmt}")

def table_more(state):
    if not _g_last_table:
        raise Signal.make(T.ValueError, state, None, "No table yet")

    return table_repr(_g_last_table, state, _g_last_offset)


def module_repr(module, state):
    res = f'<Module {module.name} | {len(module.namespace)} members>'
    if state.fmt == 'html':
        res = html.escape(res)
    return res

def function_repr(func, state):
    res = '<%s>' % func.help_str(state)
    if state.fmt == 'html':
        res = html.escape(res)
    return res


class Display:
    def print(self, repr_):
        print(repr_)

class RichDisplay(Display):
    def __init__(self):
        self.console = rich.console.Console()

    def print(self, repr_, end="\n"):
        if hasattr(repr_, '__rich_console__'):
            self.console.print(repr_, overflow="ellipsis", end=end)
        else:
            # repr_ = rich.text.Text(repr_)
            self.console.print(repr_, overflow="ellipsis", markup=False, end=end)

    def print_exception(self, e):
        "Yields colorful styled lines to print by the ``rich`` library"
        self.console.print('[bold]Exception traceback:[/bold]')
        for ref in e.text_refs:
            for line in (ref.get_pinpoint_text(rich=True) if ref else ['???']):
                self.console.print(line)
            self.console.print()
        self.console.print(rich.text.Text('%s: %s' % (e.type, e.message)))



def install_reprs():
    objects.CollectionInstance.repr = table_repr
    objects.Module.repr = module_repr
    objects.Function.repr = function_repr

display = RichDisplay()
