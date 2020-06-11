import sys
import logging

logging.basicConfig(level=logging.INFO,
                    format="(%(levelname)s) %(name)s -- %(message)s",
                    )#datefmt='%m-%d %H:%M')

optimize = True
cache = False

debug = True # not sys.flags.optimize
if debug:
    print("# DEBUG MODE")

