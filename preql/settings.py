
import sys

optimize = True
cache = False
debug = False

print_sql = False
typecheck = False # not sys.flags.optimize
autocomplete = True

try:
    from .local_settings import *
except ImportError:
    pass