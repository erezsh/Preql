import argparse
from pathlib import Path
from itertools import chain
import rich

from . import Preql, __version__, Signal

parser = argparse.ArgumentParser(description='Preql command-line interface')
parser.add_argument('-i', '--interactive', action='store_true', default=False, help="Enter interactive mode after running the script")
parser.add_argument('-v', '--version', action='store_true', help="Print version")
parser.add_argument('--install-jupyter', action='store_true', help="Installs the Preql plugin for Jupyter notebook")
parser.add_argument('--print-sql', action='store_true', help="Print the SQL code that's being executed")
parser.add_argument('-f', '--file', type=str, help='Path to a Preql script to run')
parser.add_argument('-m', '--module', type=str, help='Name of a Preql module to run')
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

    if args.install_jupyter:
        from .jup_kernel.install import main as install_jupyter
        install_jupyter([])
        print("Install successful. To start working, run 'jupyter notebook' and create a new Preql notebook.")

    p = Preql(print_sql=args.print_sql)
    if args.database:
        p.interp.state.connect(args.database)

    interactive = args.interactive

    res = 0
    try:
        if args.file:
            p.load(args.file)
        elif args.module:
            p('import ' + args.module)
        elif args.version or args.install_jupyter:
            pass
        else:
            dot_preql = find_dot_preql()
            if dot_preql:
                print("Auto-running", dot_preql)
                p._run_code(dot_preql.read_text(), dot_preql)

            interactive = True
    except Signal as e:
        for is_rich, line in e.get_rich_lines():
            if is_rich:
                rich.print(line)
            else:
                print(line)
        res = -1
    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C)")


    if interactive:
        p.load_all_tables()
        p.start_repl()
    else:
        return res

if __name__ == '__main__':
    main()