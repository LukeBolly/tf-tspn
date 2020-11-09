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
    config.learning_rate = 0.01

    # training config
    config.num_epochs = 100
    config.batch_size = 32
    return config


class Tspn:
    def __init__(self, config, dataset):
        self._c = config
        self._should_eval = Every(config.train_steps)
        self._step = 0
        self.max_set_size = dataset.max_num_elements
        self.element_size = dataset.element_size

        self.train_ds, self.val_ds, self.test_ds = dataset.get_train_val_test()

        self._size_pred = SizePredictor(self._c.size_pred_width)
        self._encoder = FSPool(self.element_size, self._c.fspool_n_pieces)
        self._prior = SetPrior(self.element_size)
        self._transformer = Encoder(self._c.transformer_depth, self._c.attn_size, self._c.num_heads,
                                    self._c.dense_size, dataset.element_size)
        self.optimiser = tf.optimizers.Adam(self._c.learning_rate)

    def train(self):
        self.train_ds = self.train_ds.batch(self._c.batch_size)
        for epoch in range(self._c.num_epochs):
            print(' starting epoch: ' + str(epoch))
            for step, (images, sets, sizes, labels) in enumerate(self.train_ds):
                self.train_set_predictor_step(sets, sizes)

        stuff = self._prior.sample(100)
        pass



    def train_set_predictor_step(self, x, sizes):
        with tf.GradientTape() as prior_tape:
            padded_x = x[:, :, 0]
            unpadded_indices = tf.where(tf.not_equal(padded_x, self._c.pad_value))
            total_elements = tf.reduce_sum(sizes)
            sampled_elements = self._prior(total_elements)  # [batch_size, max_set_size, num_features]
            initial_elements = tf.cast(tf.gather_nd(x, unpadded_indices), tf.float32)

            negloglik = lambda y, p_y: -p_y.log_prob(y)
            prior_error = negloglik(initial_elements, sampled_elements)
            prior_loss = tf.reduce_mean(prior_error)

        with tf.GradientTape() as model_tape:
            x_trans = tf.transpose(x, [0, 2, 1])
            encoded, perm = self._encoder(x_trans, sizes)   # pooled: [batch_size, num_features]
            encoded_shaped = tf.tile(tf.expand_dims(encoded, 1), [1, self.max_set_size, 1])
            # concat the conditioning vector onto each element
            samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_elements, sizes)
            padded_samples = samples_ragged.to_tensor(default_value=self._c.pad_value,
                                                      shape=[self._c.batch_size, self.max_set_size, self.element_size])
            sampled_elements_conditioned = tf.concat([padded_samples, encoded_shaped], 2)
            mask = tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, self.max_set_size)), tf.float32)
            mask = mask[:, tf.newaxis, tf.newaxis, :]   # mask shape for transformer is (batch_size, 1, 1, seq_len)
            pred_set = self._transformer(sampled_elements_conditioned, mask)

            # although the arrays contain padded values, chamfer loss is a sum over elements so it wont effect loss
            dist = chamfer_distance(x, pred_set)
            model_loss = tf.reduce_mean(dist, axis=0)

        prior_grads = prior_tape.gradient(prior_loss, self._prior.trainable_weights)
        self.optimiser.apply_gradients(zip(prior_grads, self._prior.trainable_weights))

        model_trainables = self._encoder.trainable_weights + self._transformer.trainable_weights
        model_grads = model_tape(model_loss, model_trainables)
        self.optimiser.apply_gradients(zip(model_grads, model_trainables))


    def train_size_predictor_step(self, x, sizes):
        with tf.GradientTape() as size_tape:
            pooled, perm = self._encoder(x, sizes)   # pooled: [batch_size, num_features]
            set_sizes_pred = self._size_pred(pooled)
            size_loss = tf.keras.losses.mean_squared_error(sizes, set_sizes_pred)



if __name__ == '__main__':
    config = set_config()
    dataset = MnistSet(config.pad_value)
    tspn = Tspn(config, dataset)
    tspn.train()
