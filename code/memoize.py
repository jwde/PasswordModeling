import functools

def memoize(fun):
    class memoizer(dict):
        def __init__(self, fun):
            self.fun = fun

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            self[key] = self.fun(*key)
            return self[key]

        def __get__(self, o, ot=None):
            if o is None:
                return self.fun
            fun = functools.partial(self.__call__, o)
            fun.reset = self._reset
            return fun

        def _reset(self):
            self.clear()

    return memoizer(fun)
