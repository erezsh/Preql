import inspect
from collections import defaultdict
from functools import wraps

class DispatchError(TypeError):
    pass

def ismethod(f):
    # Very hacky, but there doesn't seem to be a better test
    qname = f.__qualname__
    qname = qname.split('<locals>.')[-1]
    return '.' in qname

def get_func_simple_signature(f):
    res = []
    sig = inspect.signature(f)
    for p in sig.parameters.values():
        t = p.annotation
        if t is inspect._empty:
            t = None
        elif not isinstance(t, type):
            raise TypeError("Annotation isn't a type")
        res.append(t)
    return tuple(res)

def get_args_simple_signature(args):
    return tuple(type(a) for a in args)


class Dispatchy:
    def __init__(self):
        self._functions = defaultdict(dict)

    def __call__(self, f):
        dispatch_dict = self._functions[f.__name__]
        start_index = 1 if ismethod(f) else 0
        signature = get_func_simple_signature(f)[start_index:]
        if signature in dispatch_dict:
            raise TypeError("Collision in %s" % (signature,))
        dispatch_dict[signature] = f

        @wraps(f)
        def dispatched_f(*args, **kw):
            sig = get_args_simple_signature(args[start_index:])
            try:
                target_f = dispatch_dict[sig]
            except KeyError:
                sig_text = ', '.join([t.__name__ for t in sig])
                raise DispatchError(f"No such function {f.__name__}({sig_text})")
            return target_f(*args, **kw)



        return dispatched_f
