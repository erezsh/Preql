from logging import getLogger, Formatter, StreamHandler, basicConfig
from logging import DEBUG, INFO, WARN, ERROR, CRITICAL

basicConfig(level=INFO, format="(%(levelname)s) %(name)s -- %(message)s")
#             )#datefmt='%m-%d %H:%M')

sh = StreamHandler()
sh.setFormatter(Formatter('%(message)s'))

def make_logger(name, level):
    logger = getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    logger.addHandler(sh)
    return logger



sql_log = make_logger('sql_output', DEBUG)
ac_log = make_logger('autocomplete', CRITICAL)
repl_log = make_logger('repl', INFO)
test_log = make_logger('tests', ERROR)
