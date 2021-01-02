import tensorflow as tf
from models.size_predictor import SizePredictor
from tools import AttrDict, Every
from datasets.mnist_set import MnistSet
from models.functions.chamfer_distance import chamfer_distance
import datetime
from visualisation import mnist_visualiser
import math
from models.tspn import Tspn


def set_config():
    config = AttrDict()

    # model config
    config.train_split = 80
    config.trans_layers = 3
    config.trans_attn_size = 256
    config.trans_num_heads = 4
    config.encoder_latent = 256
    config.encoder_output_channels = 64
    config.fspool_n_pieces = 20
    config.size_pred_width = 128
    config.train_steps = 100
    config.pad_value = -1
    config.tspn_learning_rate = 0.001
    config.prior_learning_rate = 0.1
    config.weight_decay = 0.0001
    config.log_every = 500

    # training config
    config.num_epochs = 100
    config.batch_size = 32
    return config


class TspnMnist:
    def __init__(self, config, dataset):
        self._c = config
        self._should_eval = Every(config.train_steps)
        self._step = 0
        self.max_set_size = dataset.max_num_elements
        self.element_size = dataset.element_size
        self.should_log = Every(self._c.log_every)
        self.dataset = dataset

        self._size_pred = SizePredictor(self._c.size_pred_width)

        self.tspn = Tspn(self._c.encoder_latent, self._c.encoder_output_channels, self._c.fspool_n_pieces,
                         self._c.trans_layers, self._c.trans_attn_size, self._c.trans_num_heads,
                         self.dataset.element_size, self._c.pad_value, self.dataset.max_num_elements)

        self.tspn_optimiser = tf.optimizers.Adam(self._c.tspn_learning_rate)
        self.prior_optimiser = tf.optimizers.Adam(self._c.prior_learning_rate)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/metrics/' + current_time
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def train(self):
        train_ds = self.dataset.get_train_set().batch(self._c.batch_size)
        val_ds = self.dataset.get_val_set().batch(self._c.batch_size)

        self._step = 0
        # start by training our prior
        print('prior training')
        for (images, sets, sizes, labels) in train_ds.take(100):
            train_prior_loss = self.train_prior_step(sets, sizes)
            self._step += 1

            with self.summary_writer.as_default():
                tf.summary.scalar('train/prior loss', train_prior_loss, step=self._step)

        self._step = 0
        # once prior has stabilised, begin training TSPN
        for epoch in range(self._c.num_epochs):
            print('tspn training epoch: ' + str(epoch))
            for train_step, (images, sets, sizes, labels) in enumerate(train_ds):
                train_model_loss = self.train_tspn_step(sets, sizes)
                self._step += 1

                with self.summary_writer.as_default():
                    tf.summary.scalar('train/model loss', train_model_loss, step=self._step)

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

    def prior_loss(self, initial_set, sizes):
        sampled_set = self.tspn.sample_prior(sizes)

        # exclude padded values and flatten our batch of sets
        unpadded_indices = tf.where(tf.not_equal(initial_set[:, :, 0], self._c.pad_value))
        initial_set_flattened = tf.cast(tf.gather_nd(initial_set, unpadded_indices), tf.float32)

        negloglik = lambda y, p_y: -p_y.log_prob(y)
        prior_error = negloglik(initial_set_flattened, sampled_set)
        prior_loss = tf.reduce_mean(prior_error)

        samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_set, sizes)
        padded_samples = samples_ragged.to_tensor(default_value=self._c.pad_value,
                                                  shape=[sizes.shape[0], self.max_set_size, self.element_size])

        return padded_samples, prior_loss

    def train_prior_step(self, x, sizes):
        with tf.GradientTape() as prior_tape:
            sampled_set, prior_loss = self.prior_loss(x, sizes)

        prior_trainables = self.tspn.get_prior_weights()
        prior_grads = prior_tape.gradient(prior_loss, prior_trainables)
        self.prior_optimiser.apply_gradients(zip(prior_grads, prior_trainables))

        return prior_loss

    def tspn_loss(self, x, sampled_set, sizes):
        pred_set = self.tspn(x, sampled_set, sizes)

        # although the arrays contain padded values, chamfer loss is a sum over elements so it wont effect loss
        dist = chamfer_distance(x, pred_set, sizes)
        model_loss = tf.reduce_mean(dist, axis=0)

        return pred_set, model_loss

    def train_tspn_step(self, initial_set, sizes):
        sampled_set = self.tspn.sample_prior_batch(sizes)

        with tf.GradientTape() as model_tape:
            pred_set, model_loss = self.tspn_loss(initial_set, sampled_set, sizes)

        model_trainables = self.tspn.get_autoencoder_weights()
        model_grads = model_tape.gradient(model_loss, model_trainables)
        self.tspn_optimiser.apply_gradients(zip(model_grads, model_trainables))
        return model_loss

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
    dataset = MnistSet(config.train_split, config.pad_value, 100)
    tspn = TspnMnist(config, dataset)
    tspn.train()
