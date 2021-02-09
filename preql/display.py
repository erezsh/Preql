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

from .context import context

TABLE_PREVIEW_SIZE = 16
LIST_PREVIEW_SIZE = 128
MAX_AUTO_COUNT = 10000

@dp_type
def pql_repr(t: T.function, value):
    params = []
    for p in value.params:
        s = p.name
        if p.type:
            s += ": %s" % p.type
        params.append(s)

    return f'{{func {value.name}({", ".join(params)})}}'

@dp_type
def pql_repr(t: T.decimal, value):
    raise Signal.make(T.NotImplementedError, None, "Decimal not implemented")

@dp_type
def pql_repr(t: T.string, value):
    assert isinstance(value, str), value
    value = value.replace('"', r'\"')
    res = f'"{value}"'
    if context.state.fmt == 'html':
        res = html.escape(res)
    return res

@dp_type
def pql_repr(t: T.text, value):
    assert isinstance(value, str), value
    return str(value)

@dp_type
def pql_repr(t: T._rich, value):
    return rich.text.Text.from_markup(str(value))

@dp_type
def pql_repr(t: T.bool, value):
    return 'true' if value else 'false'

@dp_type
def pql_repr(t: T.nulltype, value):
    return 'null'




def table_limit(table, state, limit, offset=0):
    return call_pql_func(state, 'limit_offset', [table, ast.make_const(limit), ast.make_const(offset)])

def _call_pql_func(state, name, args):
    count = call_pql_func(state, name, args)
    return cast_to_python(state, count)

def _html_table(name, count_str, rows, offset, has_more, colors):
    assert colors
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

    if has_more:
        trs.append('<tr><td>...</td></tr>')

    style = """<style>
    .preql_table td, .preql_table th {
        text-align: left
    }
    </style>
    """
    return '%s<table class="preql_table">%s%s</table>' % (header, ths, '\n'.join(trs)) + style


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

def _view_table(state, table, size, offset):
    global _g_last_table, _g_last_offset
    rows = cast_to_python(state, table_limit(table, state, size, offset))
    _g_last_table = table
    _g_last_offset = offset + len(rows)
    if table.type <= T.list:
        rows = [{ITEM_NAME: x} for x in rows]

    try:
        table_name = table.type.options['name'].repr_name
    except KeyError:
        table_name = ''

    return table_name, rows


def table_repr(self, offset=0):
    state = context.state

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
    table_f = _rich_table
    preview = TABLE_PREVIEW_SIZE
    colors = True

    if state.fmt == 'html':
        preview = TABLE_PREVIEW_SIZE * 10
        table_f = _html_table
    elif state.fmt == 'text':
        colors = False
    else:
        assert state.fmt == 'rich'

    table_name, rows, = _view_table(state, self, preview, offset)
    has_more = offset + len(rows) < count
    return table_f(table_name, count_str, rows, offset, has_more, colors=colors)

    # raise NotImplementedError(f"Unknown format: {state.fmt}")

def table_more():
    if not _g_last_table:
        raise Signal.make(T.ValueError, None, "No table yet")

    return table_repr(_g_last_table, _g_last_offset)


def module_repr(module):
    res = f'<Module {module.name} | {len(module.namespace)} members>'
    if context.state.fmt == 'html':
        res = html.escape(res)
    return res

def function_repr(func):
    res = '<%s>' % func.help_str()
    if context.state.fmt == 'html':
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
