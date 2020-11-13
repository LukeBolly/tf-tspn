# useful tools taken from Tensorflow Dreamer repository


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Every:
    def __init__(self, every):
        self._every = every
        self._last = 0

    def __call__(self, step):
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False
