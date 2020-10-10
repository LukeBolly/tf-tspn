import tensorflow as tf
from models.set_prior import SetPrior
from models.transformer import Transformer
from tools import AttrDict


def set_config():
    config = AttrDict()
    config.transformer_depth = 4
    config.dense_size = 512
    config.attn_size = 128
    config.num_heads = 4


class Tspn():
    def __init__(self, config):
        self._c = config

    def build_models(self):
        self._prior = SetPrior(self._c)
        self._model = Transformer(self._c.transformer_depth, self._c.attn_size, self._c.num_heads, self._c.dense_size)


if __name__ == '__main__':
    Tspn()
