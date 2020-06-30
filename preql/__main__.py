import argparse
from . import Preql, __version__
from pathlib import Path
from itertools import chain

parser = argparse.ArgumentParser(description='Preql command-line interface')
parser.add_argument('-i', '--interactive', action='store_true', help="Enter interactive mode after running the script")
parser.add_argument('-v', '--version', action='store_true', help="Print version")
parser.add_argument('--print-sql', action='store_true', help="Print the SQL code that's being executed")
parser.add_argument('script_path', type=str, nargs='?', default=None, help='Path to a Preql script to run')

def find_dot_preql():
    cwd = Path.cwd()
    for p in chain([cwd], cwd.parents):
        dot_preql = p / ".preql"
        if dot_preql.exists():
            return dot_preql

def main():
    args = parser.parse_args()

    if args.version:
        print(__version__)

    p = Preql(print_sql=args.print_sql)

    interactive = args.interactive

    if args.script_path:
        with open(args.script_path) as f:
            p.run_code(f.read(), args.script_path)
    elif not args.version:
        dot_preql = find_dot_preql()
        if dot_preql:
            print("Auto-running", dot_preql)
            p.run_code(dot_preql.read_text(), dot_preql)

            interactive = True

    if interactive:
        p.start_repl()

if __name__ == '__main__':
    main()