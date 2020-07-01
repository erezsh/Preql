import argparse
from . import Preql, __version__
from pathlib import Path
from itertools import chain

parser = argparse.ArgumentParser(description='Preql command-line interface')
parser.add_argument('-i', '--interactive', action='store_true', default=False, help="Enter interactive mode after running the script")
parser.add_argument('-v', '--version', action='store_true', help="Print version")
parser.add_argument('--print-sql', action='store_true', help="Print the SQL code that's being executed")
parser.add_argument('-f', '--file', type=str, help='Path to a Preql script to run')
parser.add_argument('database', type=str, nargs='?', default=None, help="database url (postgres://user:password@host:port/db_name")

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
    if args.database:
        p.interp.state.connect(args.database)

    interactive = args.interactive

    if args.file:
        with open(args.file) as f:
            p.run_code(f.read(), args.file)
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