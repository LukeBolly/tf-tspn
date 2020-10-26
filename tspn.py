import tensorflow as tf
from models.set_prior import SetPrior
from models.transformer import Encoder
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
    config.max_set_size = 400
    config.pad_value = -99999
    return config


class Tspn(tf.keras.Model):
    def __init__(self, config, dataset):
        super().__init__()
        self._c = config
        self._should_eval = Every(config.train_steps)
        self._step = 0

        self.train_ds, self.val_ds, self.test_ds = dataset.get_train_val_test(self._c.pad_value)

        self._size_pred = SizePredictor(self._c.size_pred_width, )
        self._fsPool = FSPool(dataset.element_size, self._c.fspool_n_pieces)
        self._prior = SetPrior(dataset.element_size, self._c.max_set_size)
        self._transformer = Encoder(self._c.transformer_depth, self._c.attn_size, self._c.num_heads,
                                    self._c.dense_size, rate=0)

    def train_step(self, x, sizes):
        with tf.GradientTape() as size_tape:
            unpadded_indices = tf.where(tf.not_equal(x, self._c.pad_value))
            pooled, perm = self._fsPool(x, sizes)   # pooled: [batch_size, num_features]
            set_sizes_pred = self._size_pred(pooled)
            size_loss = tf.keras.losses.mean_squared_error(sizes, set_sizes_pred)


        with tf.GradientTape() as prior_tape:
            padded_sampled_elements = self._prior()  # [batch_size, max_set_size, num_features]
            initial_elements = tf.gather_nd(x, unpadded_indices)
            sampled_elements = tf.gather_nd(padded_sampled_elements, unpadded_indices)

            negloglik = lambda y, p_y: -p_y.log_prob(y)
            prior_loss = negloglik(initial_elements, sampled_elements)

        with tf.GradientTape() as model_tape:
            # concat the conditioning vector onto each element
            sampled_elements_conditioned = tf.concat([padded_sampled_elements, pooled], 2)
            mask = tf.sequence_mask(sizes, self._c.max_set_size)
            pred_set = self._transformer(sampled_elements_conditioned, True, mask)

        size_grad = size_tape.gradient(size_loss, self._prior.trainable_weights)




if __name__ == '__main__':
    config = set_config()
    dataset = MnistSet()
    Tspn(config, dataset)
