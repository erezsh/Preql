import argparse
from . import Preql
from pathlib import Path

parser = argparse.ArgumentParser(description='Preql command-line interface')
parser.add_argument('-i', '--interactive', action='store_true', help="Enter interactive mode after running the script")
parser.add_argument('-d', '--debug', action='store_true', help="Display debug information")
parser.add_argument('-f', '--file', type=str, help='Path to a Preql script to run')
parser.add_argument('database', type=str, nargs='?', default=None, help="database url (postgres://user:password@host:port/db_name")

def find_dot_preql():
    current_dir = Path.cwd()
    while True:
        dot_preql = current_dir / ".preql"
        if dot_preql.exists():
            return dot_preql

        if current_dir.is_mount():
            return

        current_dir = current_dir.parent



def main():
    args = parser.parse_args()

    p = Preql(debug=args.debug)
    if args.database:
        p.interp.state.connect(args.database)

    interactive = args.interactive

    if args.file:
        with open(args.file) as f:
            p.run_code(f.read(), args.file)
    else:
        dot_preql = find_dot_preql()
        if dot_preql:
            p.run_code(dot_preql.read_text(), dot_preql)

        interactive = True

    if interactive:
        p.start_repl()

if __name__ == '__main__':
    main()