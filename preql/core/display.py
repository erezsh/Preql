import html

import rich.table
import rich.text
import rich.console

from .exceptions import Signal
from .pql_types import T, ITEM_NAME
from . import pql_objects as objects
from .pql_ast import pyvalue
from .types_impl import dp_type, pql_repr
from .interp_common import call_builtin_func, cast_to_python_int, cast_to_python
from .state import get_display


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
    if not isinstance(value, str):
        raise Signal.make(T.TypeError, None, f"Expected value of type 'string', instead got {type(value)}")

    value = value.replace('"', r'\"')
    res = f'"{value}"'
    if get_display().format == 'html':
        res = html.escape(res)
    return res

@dp_type
def pql_repr(t: T.text, value):
    assert isinstance(value, str), value
    return str(value)

@dp_type
def pql_repr(t: T._rich, value):
    r = rich.text.Text.from_markup(str(value))
    if get_display().format == 'html':
        return _rich_to_html(r)
    return r

@dp_type
def pql_repr(t: T.bool, value):
    return 'true' if value else 'false'

@dp_type
def pql_repr(t: T.nulltype, value):
    return 'null'


def _rich_to_html(r):
    console = rich.console.Console(record=True)
    console.print(r)
    return console.export_html(code_format='<style>{stylesheet}</style><pre>{code}</pre>').replace('━', '-')


def table_limit(table, limit, offset=0):
    assert isinstance(limit, int)
    assert isinstance(offset, int)
    return call_builtin_func('limit_offset', [table, pyvalue(limit), pyvalue(offset)])


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

def _view_table(table, size, offset):
    global _g_last_table, _g_last_offset
    rows = cast_to_python(table_limit(table, size, offset))
    _g_last_table = table
    _g_last_offset = offset + len(rows)
    if table.type <= T.list:
        rows = [{ITEM_NAME: x} for x in rows]

    try:
        table_name = table.type.options['name'].repr_name
    except KeyError:
        table_name = ''

    return table_name, rows


def table_inline_repr(self):
    offset = 0
    table_name, rows, = _view_table(self, TABLE_PREVIEW_SIZE, offset)
    return '[%s]' % ', '.join(repr(r) for r in rows)



def table_repr(self, offset=0):

    count = cast_to_python_int(call_builtin_func('count', [table_limit(self, MAX_AUTO_COUNT)]))
    if count == MAX_AUTO_COUNT:
        count_str = f'>={count}'
    else:
        count_str = f'={count}'

    # if len(self.type.elems) == 1:
    #     rows = cast_to_python(table_limit(self, LIST_PREVIEW_SIZE))
    #     post = f', ... ({count_str})' if len(rows) < count else ''
    #     elems = ', '.join(repr_value(ast.Const(None, self.type.elem, r)) for r in rows)
    #     return f'[{elems}{post}]'

    # TODO load into preql and repr, instead of casting to python
    table_f = _rich_table
    preview = TABLE_PREVIEW_SIZE
    colors = True
    display = get_display()

    if display.format == 'html':
        preview = TABLE_PREVIEW_SIZE * 10
        table_f = _html_table
    elif display.format == 'text':
        colors = False
    else:
        assert display.format == 'rich'

    table_name, rows, = _view_table(self, preview, offset)
    has_more = offset + len(rows) < count
    return table_f(table_name, count_str, rows, offset, has_more, colors=colors)


def table_more():
    if not _g_last_table:
        raise Signal.make(T.ValueError, None, "No table yet")

    return table_repr(_g_last_table, _g_last_offset)


def module_repr(module):
    res = f'<Module {module.name} | {len(module.namespace)} members>'
    if get_display().format == 'html':
        res = html.escape(res)
    return res

def function_repr(func):
    res = '<%s>' % func.help_str()
    if get_display().format == 'html':
        res = html.escape(res)
    return res


class Display:
    def print(self, repr_):
        print(repr_)

def _print_rich_exception(console, e):
    console.print('[bold]Exception traceback:[/bold]')
    for ref in e.text_refs:
        for line in (ref.get_pinpoint_text(rich=True) if ref else ['???']):
            console.print(line)
        console.print()
    console.print(rich.text.Text('%s: %s' % (e.type, e.message)))

class RichDisplay(Display):
    format = "rich"

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
        _print_rich_exception(self.console, e)


def print_to_string(x, format):
    console = rich.console.Console(color_system=None)
    with console.capture() as capture: 
        console.print(x)
    return capture.get()


class HtmlDisplay(Display):
    format = "html"

    def __init__(self):
        # self.console = rich.console.Console(record=True)
        self.buffer = []

    def print(self, repr_, end="<br/>"):
        self.buffer.append(str(repr_) + end)

    # print = RichDisplay.print
    def print_exception(self, e):
        console = rich.console.Console(record=True)
        _print_rich_exception(console, e)
        res = console.export_html(code_format='<style>{stylesheet}</style><pre>{code}</pre>').replace('━', '-')
        self.buffer.append(res)

    def as_html(self):
        # return '\n'.join(self.buffer) + "%%"
        # return self.console.export_html(code_format='<style>{stylesheet}</style><pre>{code}</pre>').replace('━', '-')
        res = '\n'.join(self.buffer)
        self.buffer.clear()
        return res



def install_reprs():
    objects.CollectionInstance.repr = table_repr
    objects.CollectionInstance.inline_repr = table_inline_repr
    objects.Module.repr = module_repr
    objects.Function.repr = function_repr
