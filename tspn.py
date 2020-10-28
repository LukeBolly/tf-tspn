import tensorflow as tf
from models.set_prior import SetPrior
from models.transformer import Encoder
from models.fspool import FSPool
from models.size_predictor import SizePredictor
from tools import AttrDict, Every
from datasets.mnist_set import MnistSet
from models.functions.chamfer_distance import chamfer_distance


def set_config():
    config = AttrDict()

    # model config
    config.transformer_depth = 4
    config.dense_size = 512
    config.attn_size = 128
    config.num_heads = 4
    config.fspool_n_pieces = 20
    config.size_pred_width = 128
    config.train_steps = 100
    config.pad_value = -99999

    # training config
    config.num_epochs = 10
    config.batch_size = 100
    return config


class Tspn(tf.keras.Model):
    def __init__(self, config, dataset):
        super().__init__()
        self._c = config
        self._should_eval = Every(config.train_steps)
        self._step = 0

        self.train_ds, self.val_ds, self.test_ds = dataset.get_train_val_test(self._c.pad_value)

        self._size_pred = SizePredictor(self._c.size_pred_width)
        self._fsPool = FSPool(dataset.element_size, self._c.fspool_n_pieces)
        self._prior = SetPrior(self._c.batch_size, dataset.max_num_elements, dataset.element_size)
        self._transformer = Encoder(self._c.transformer_depth, self._c.attn_size, self._c.num_heads,
                                    self._c.dense_size, rate=0)

    def train(self):
        self.train_ds = self.train_ds.batch(self._c.batch_size)
        for epoch in range(self._c.num_epochs):
            print(' starting epoch: ' + str(epoch))
            for step, (images, sets, sizes, labels) in enumerate(self.train_ds):
                self.train_set_predictor_step(sets, sizes)


    def train_set_predictor_step(self, x, sizes):
        with tf.GradientTape() as prior_tape:
            unpadded_indices = tf.where(tf.not_equal(x, self._c.pad_value))
            padded_sampled_elements = self._prior()  # [batch_size, max_set_size, num_features]
            initial_elements = tf.gather_nd(x, unpadded_indices)
            sampled_elements = tf.gather_nd(padded_sampled_elements, unpadded_indices)

            negloglik = lambda y, p_y: -p_y.log_prob(y)
            prior_loss = negloglik(initial_elements, sampled_elements)

        with tf.GradientTape() as model_tape:
            pooled, perm = self._fsPool(x, sizes)   # pooled: [batch_size, num_features]
            # concat the conditioning vector onto each element
            sampled_elements_conditioned = tf.concat([padded_sampled_elements, pooled], 2)
            mask = tf.sequence_mask(sizes, self._c.max_set_size)
            pred_set = self._transformer(sampled_elements_conditioned, True, mask)

            # although the arrays contain padded values, chamfer loss is a sum over elements so it wont effect loss
            dist = chamfer_distance(x, pred_set)
            model_loss = tf.reduce_mean(dist, axis=0)

        size_grad = size_tape.gradient(size_loss, self._prior.trainable_weights)

    def train_size_predictor_step(self, x, sizes):
        with tf.GradientTape() as size_tape:
            pooled, perm = self._fsPool(x, sizes)   # pooled: [batch_size, num_features]
            set_sizes_pred = self._size_pred(pooled)
            size_loss = tf.keras.losses.mean_squared_error(sizes, set_sizes_pred)



if __name__ == '__main__':
    config = set_config()
    dataset = MnistSet()
    Tspn(config, dataset)
