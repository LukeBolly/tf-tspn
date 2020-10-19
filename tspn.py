import tensorflow as tf
from models.set_prior import SetPrior
from models.transformer import Transformer
from models.fspool import FSPool
from models.size_predictor import SizePredictor
from tools import AttrDict, Every
from datasets.mnist_set import MnistSet


def set_config():
    config = AttrDict()
    config.transformer_depth = 4
    config.dense_size = 512
    config.attn_size = 128
    config.num_heads = 4
    config.fspool_n_pieces = 20
    config.size_pred_width = 128
    config.train_steps = 100
    return config


class Tspn(tf.keras.Model):
    def __init__(self, config, dataset):
        super().__init__()
        self._c = config
        self._should_eval = Every(config.train_steps)
        self._step = 0

        self.train_ds, self.val_ds, self.test_ds = dataset.get_train_val_test()

        self._size_pred = SizePredictor(self._c.size_pred_width, )
        self._fsPool = FSPool(dataset.element_size, self._c.fspool_n_pieces)
        self._prior = SetPrior(dataset.element_size)
        self._transformer = Transformer(self._c.transformer_depth, self._c.attn_size, self._c.num_heads,
                                        self._c.dense_size, dataset.element_size, rate=0)

    def train_and_test(self, x, sizes):
        if self._should_eval(self._step):
            pass
        else:
            with tf.GradientTape() as input_tape:
                pooled, perm = self._fsPool(x, sizes)
                cardinality = self._size_pred(pooled)

            with tf.GradientTape() as prior_tape:
                initial_elements = self._prior(None)





if __name__ == '__main__':
    config = set_config()
    dataset = MnistSet()
    Tspn(config, dataset)
