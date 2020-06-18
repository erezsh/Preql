import argparse
from . import Preql
from pathlib import Path
from itertools import chain

parser = argparse.ArgumentParser(description='Preql command-line interface')
parser.add_argument('-i', '--interactive', action='store_true', help="Enter interactive mode after running the script")
parser.add_argument('-d', '--debug', action='store_true', help="Display debug information")
parser.add_argument('script_path', type=str, nargs='?', default=None, help='Path to a Preql script to run')

def find_dot_preql():
    cwd = Path.cwd()
    for p in chain([cwd], cwd.parents):
        dot_preql = p / ".preql"
        if dot_preql.exists():
            return dot_preql

def main():
    args = parser.parse_args()

    p = Preql(debug=args.debug)

    interactive = args.interactive

    if args.script_path:
        with open(args.script_path) as f:
            p.run_code(f.read(), args.script_path)
    else:
        dot_preql = find_dot_preql()
        if dot_preql:
            p.run_code(dot_preql.read_text(), dot_preql)

        interactive = True

    if interactive:
        p.start_repl()

if __name__ == '__main__':
    main()