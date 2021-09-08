
import sys

optimize = True
cache = False
debug = False

print_sql = False
typecheck = False # not sys.flags.optimize
autocomplete = True

color_theme = {
    'text'    : '#c0c0c0',
    'comment' : 'italic #808080',
    'keyword' : 'bold #0060f0',
    'name'    : '#f0f0f0',
    'name_func'  : 'bold #f0f0f0',
    'name_class' : 'bold #f0f0f0',
    'string' : '#40f040',
    'number' : '#40f0f0',
    'operator' : '#c0c0c0',
    'error' : '#f04040',
}

update_color_theme = {}

class Display:
    TABLE_PREVIEW_SIZE_SHELL = 16
    TABLE_PREVIEW_SIZE_HTML = 64
    LIST_PREVIEW_SIZE = 128
    MAX_AUTO_COUNT = 10000

try:
    from .local_settings import *
except ImportError:
    pass


color_theme.update(update_color_theme)