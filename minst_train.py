import tensorflow as tf
from models.set_prior import SetPrior
from models.transformer import Encoder
from models.fspool import FSEncoder
from models.size_predictor import SizePredictor
from tools import AttrDict, Every
from datasets.mnist_set import MnistSet
from models.functions.chamfer_distance import chamfer_distance
import datetime
from visualisation import mnist_visualiser
import math


def set_config():
    config = AttrDict()

    # model config
    config.train_split = 80
    config.transformer_depth = 4
    config.dense_size = 512
    config.attn_size = 128
    config.num_heads = 4
    config.encoder_dim = 64
    config.latent_dim = 8
    config.fspool_n_pieces = 20
    config.size_pred_width = 128
    config.train_steps = 100
    config.pad_value = -99999
    config.tspn_learning_rate = 0.0001
    config.prior_learning_rate = 0.1
    config.log_every = 500

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
        self.should_log = Every(self._c.log_every)
        self.dataset = dataset

        self._size_pred = SizePredictor(self._c.size_pred_width)
        self._encoder = FSEncoder(self._c.encoder_dim, self._c.latent_dim, self._c.fspool_n_pieces)
        self._prior = SetPrior(self.element_size)
        self._transformer = Encoder(self._c.transformer_depth, self._c.attn_size, self._c.num_heads,
                                    self._c.dense_size, dataset.element_size)
        self.tspn_optimiser = tf.optimizers.Adam(self._c.tspn_learning_rate)
        self.prior_optimiser = tf.optimizers.Adam(self._c.prior_learning_rate)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/metrics/' + current_time
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def train(self):
        train_ds = self.dataset.get_train_set().batch(self._c.batch_size)
        val_ds = self.dataset.get_val_set().batch(self._c.batch_size)

        for epoch in range(self._c.num_epochs):
            print('starting epoch: ' + str(epoch))
            for train_step, (images, sets, sizes, labels) in enumerate(train_ds):
                train_prior_loss, train_model_loss = self.train_tspn_step(sets, sizes)
                self._step += 1

                with self.summary_writer.as_default():
                    tf.summary.scalar('train/prior loss', train_prior_loss, step=self._step)
                    tf.summary.scalar('train/model loss', math.log10(train_model_loss), step=self._step)

                if self.should_log(self._step):
                    print('logging ' + str(self._step))
                    for images, sets, sizes, labels in val_ds.take(1):
                        val_prior_loss, val_model_loss, sampled_elements, pred_set = self.eval_tspn_step(sets, sizes)
                        with self.summary_writer.as_default():
                            tf.summary.image("Training data", mnist_visualiser.plot_to_image(
                                mnist_visualiser.set_to_plot(images, sets, sampled_elements, pred_set)), step=self._step)

            val_prior_loss_sum = 0
            val_model_loss_sum = 0

            for val_step, (images, sets, sizes, labels) in enumerate(val_ds):
                val_prior_loss, val_model_loss, sampled_elements, pred_set = self.eval_tspn_step(sets, sizes)
                val_prior_loss_sum = val_prior_loss_sum + val_prior_loss
                val_model_loss_sum = val_model_loss_sum + val_model_loss

            with self.summary_writer.as_default():
                tf.summary.scalar('val/prior loss', val_prior_loss_sum / val_step, step=self._step)
                tf.summary.scalar('val/model loss', math.log10(val_model_loss_sum / val_step), step=self._step)

    def run_prior(self, sizes):
        total_elements = tf.reduce_sum(sizes)
        sampled_elements = self._prior(total_elements)  # [batch_size, max_set_size, num_features]
        return sampled_elements

    def prior_loss(self, x, sizes):
        sampled_elements = self.run_prior(sizes)

        padded_x = x[:, :, 0]
        unpadded_indices = tf.where(tf.not_equal(padded_x, self._c.pad_value))
        initial_elements = tf.cast(tf.gather_nd(x, unpadded_indices), tf.float32)

        negloglik = lambda y, p_y: -p_y.log_prob(y)
        prior_error = negloglik(initial_elements, sampled_elements)
        prior_loss = tf.reduce_mean(prior_error)

        samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_elements, sizes)
        padded_samples = samples_ragged.to_tensor(default_value=self._c.pad_value,
                                                  shape=[self._c.batch_size, self.max_set_size, self.element_size])
        return padded_samples, prior_loss

    def run_tspn(self, x, padded_samples, sizes):
        encoded = self._encoder(x, sizes)  # pooled: [batch_size, num_features]
        encoded_shaped = tf.tile(tf.expand_dims(encoded, 1), [1, self.max_set_size, 1])
        # concat the conditioning vector onto each element

        sampled_elements_conditioned = tf.concat([padded_samples, encoded_shaped], 2)

        masked_values = tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, self.max_set_size)), tf.float32)
        pred_set = self._transformer(sampled_elements_conditioned, masked_values)
        return pred_set

    def tspn_loss(self, x, sampled_elements, sizes):
        pred_set = self.run_tspn(x, sampled_elements, sizes)

        # although the arrays contain padded values, chamfer loss is a sum over elements so it wont effect loss
        dist = chamfer_distance(x, pred_set, sizes)
        model_loss = tf.reduce_mean(dist, axis=0)
        if model_loss > 0.1 and self._step > 100:
            with self.summary_writer.as_default():
                tf.summary.image("Error data", mnist_visualiser.plot_to_image(
                    mnist_visualiser.set_to_plot(None, x, sampled_elements, pred_set)), step=self._step)
        return pred_set, model_loss

    def train_tspn_step(self, x, sizes):
        with tf.GradientTape() as prior_tape:
            padded_samples, prior_loss = self.prior_loss(x, sizes)

        with tf.GradientTape() as model_tape:
            pred_set, model_loss = self.tspn_loss(x, padded_samples, sizes)

        prior_grads = prior_tape.gradient(prior_loss, self._prior.trainable_weights)
        self.prior_optimiser.apply_gradients(zip(prior_grads, self._prior.trainable_weights))

        model_trainables = self._encoder.trainable_weights + self._transformer.trainable_weights
        model_grads = model_tape.gradient(model_loss, model_trainables)
        self.tspn_optimiser.apply_gradients(zip(model_grads, model_trainables))
        return prior_loss, model_loss

    def eval_tspn_step(self, x, sizes):
        padded_samples, prior_loss = self.prior_loss(x, sizes)
        pred_set, model_loss = self.tspn_loss(x, padded_samples, sizes)
        return prior_loss, model_loss, padded_samples, pred_set


    def train_size_predictor_step(self, x, sizes):
        with tf.GradientTape() as size_tape:
            pooled, perm = self._encoder(x, sizes)   # pooled: [batch_size, num_features]
            set_sizes_pred = self._size_pred(pooled)
            size_loss = tf.keras.losses.mean_squared_error(sizes, set_sizes_pred)



if __name__ == '__main__':
    config = set_config()
    dataset = MnistSet(config.train_split, config.pad_value)
    tspn = Tspn(config, dataset)
    tspn.train()
