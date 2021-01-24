import tensorflow as tf
from tools import AttrDict, Every
from datasets.mnist_set import MnistSet
from models.functions.chamfer_distance import chamfer_distance_smoothed
import datetime
from visualisation import mnist_example
import math
from models.tspn import Tspn
import argparse
import os
import decimal
from re import sub


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
    config.set_pred_learning_rate = 0.001
    config.weight_decay = 0.0001
    config.log_every = 500

    # training config
    config.num_epochs = 100
    config.batch_size = 32
    return config


class TspnAutoencoder:
    def __init__(self, load_step, config, dataset):
        self._c = config
        self._should_eval = Every(config.train_steps)
        self.max_set_size = dataset.max_num_elements
        self.element_size = dataset.element_size
        self.should_log = Every(self._c.log_every)
        self.dataset = dataset

        self.tspn = Tspn(self._c.encoder_latent, self._c.encoder_output_channels, self._c.fspool_n_pieces,
                         self._c.trans_layers, self._c.trans_attn_size, self._c.trans_num_heads,
                         self.dataset.element_size, self._c.size_pred_width, self._c.pad_value,
                         self.dataset.max_num_elements)
        self.tspn.compile()

        self.tspn_optimiser = tf.keras.optimizers.Adam(self._c.tspn_learning_rate)
        self.prior_optimiser = tf.keras.optimizers.Adam(self._c.prior_learning_rate)
        self.size_pred_optimiser = tf.keras.optimizers.Adam(self._c.set_pred_learning_rate)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/metrics/' + current_time
        checkpoint_folder = 'logs/checkpoints/'
        self.checkpoint_dir = checkpoint_folder + current_time + '/'
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

        # if step is set, try to find the step in the latest training run folder and load weights from there
        if load_step is not None:
            self.tspn.built = True

            def extract_sortable_value(value):
                first_value = sub(r"\D", "", value)
                return decimal.Decimal(first_value)

            run_folders = [f.path for f in os.scandir(checkpoint_folder) if f.is_dir()]
            run_folders = sorted(run_folders, key=extract_sortable_value)
            latest_run = run_folders[-1]

            step_ckpnts = [f.path for f in os.scandir(latest_run) if f.is_dir()]

            if load_step is -1:
                step_ckpnts = sorted(step_ckpnts, key=extract_sortable_value)
                step_folder = step_ckpnts[-1]
            else:
                step_folder = [x for x in step_ckpnts if str(load_step) in x][0]

            self.tspn.load_weights(step_folder + '/').expect_partial()

    def train_tspn(self):
        train_ds = self.dataset.get_train_set().batch(self._c.batch_size)
        val_ds = self.dataset.get_val_set().batch(self._c.batch_size)

        step = 0
        # start by training our prior
        print('prior training')
        for (images, sets, sizes, labels) in train_ds.take(100):
            train_prior_loss = self.train_prior_step(sets, sizes)
            step += 1

            with self.summary_writer.as_default():
                tf.summary.scalar('train/prior loss', train_prior_loss, step=step)

        step = 0
        # once prior has stabilised, begin training TSPN
        for epoch in range(self._c.num_epochs):
            print('tspn training epoch: ' + str(epoch))
            for train_step, (images, sets, sizes, labels) in enumerate(train_ds):
                train_model_loss = self.train_tspn_step(sets, sizes)
                step += 1

                with self.summary_writer.as_default():
                    tf.summary.scalar('train/model loss', train_model_loss, step=step)

                if self.should_log(step):
                    print('logging ' + str(step))
                    self.tspn.save_weights(self.checkpoint_dir + '/' + str(step) + '/')

                    with self.summary_writer.as_default():
                        for tf_var in self.tspn.trainable_weights:
                            tf.summary.histogram(tf_var.name, tf_var.numpy(), step=step)

                    for images, sets, sizes, labels in val_ds.take(1):
                        val_prior_loss, val_model_loss, sampled_elements, pred_set = self.eval_tspn_step(sets, sizes)
                        with self.summary_writer.as_default():
                            tf.summary.image("Training data", mnist_example.plot_to_image(
                                mnist_example.set_to_plot(images, sets, sampled_elements, pred_set)), step=step)

            val_prior_loss_sum = 0
            val_model_loss_sum = 0

            for val_step, (images, sets, sizes, labels) in enumerate(val_ds):
                val_prior_loss, val_model_loss, sampled_elements, pred_set = self.eval_tspn_step(sets, sizes)
                val_prior_loss_sum += val_prior_loss
                val_model_loss_sum += val_model_loss

            with self.summary_writer.as_default():
                tf.summary.scalar('val/prior loss', val_prior_loss_sum / val_step, step=step)
                tf.summary.scalar('val/model loss', val_model_loss_sum / val_step, step=step)

    def train_size_predictor(self):
        train_ds = self.dataset.get_train_set().batch(self._c.batch_size)
        val_ds = self.dataset.get_val_set().batch(self._c.batch_size)

        step = 0
        for epoch in range(self._c.num_epochs):
            print('size predictor training epoch: ' + str(epoch))
            for train_step, (images, sets, sizes, labels) in enumerate(train_ds):
                train_model_loss = self.train_size_predictor_step(sets, sizes)
                step += 1

                with self.summary_writer.as_default():
                    tf.summary.scalar('train/size predictor loss', train_model_loss, step=step)

                if self.should_log(step):
                    print('logging ' + str(step))
                    self.tspn.save_weights(self.checkpoint_dir + '/' + str(step) + '/')

            val_model_SE_sum = 0
            val_count = 0

            for val_step, (images, sets, sizes, labels) in enumerate(val_ds):
                predicted_sizes, val_model_loss = self.eval_size_predictor_step(sets, sizes)
                for i in range(len(predicted_sizes)):
                    val_count += 1
                    val_model_SE_sum += math.pow(predicted_sizes[i] - sizes[i], 2)

            rmse = math.sqrt(val_model_SE_sum / val_count)

            with self.summary_writer.as_default():
                tf.summary.scalar('val/size predictor RMSE', rmse, step=step)

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
        dist = chamfer_distance_smoothed(x, pred_set, sizes)
        model_loss = tf.reduce_mean(dist, axis=0)

        return pred_set, model_loss

    # @tf.function
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

    def size_predictor_loss(self, embedded_sets, sizes):
        pred_sizes = self.tspn.predict_size(embedded_sets)

        one_hot_sizes = tf.one_hot(sizes - 1, self.max_set_size)    # decrement indices by 1 as not sets are size 0
        size_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(one_hot_sizes, pred_sizes))
        predicted_sizes = tf.cast(tf.argmax(pred_sizes, 1), tf.int32) + 1
        return predicted_sizes, size_loss

    # @tf.function
    def train_size_predictor_step(self, initial_sets, sizes):
        embedded_sets = self.tspn.encode_set(initial_sets, sizes)  # pooled: [batch_size, num_features]

        with tf.GradientTape() as size_tape:
            set_sizes_pred, size_loss = self.size_predictor_loss(embedded_sets, sizes)

        size_trainables = self.tspn.get_size_predictor_weights()
        model_grads = size_tape.gradient(size_loss, size_trainables)
        self.tspn_optimiser.apply_gradients(zip(model_grads, size_trainables))
        return size_loss

    def eval_size_predictor_step(self, initial_sets, sizes):
        embedded_sets = self.tspn.encode_set(initial_sets, sizes)  # pooled: [batch_size, num_features]
        set_sizes_pred, size_loss = self.size_predictor_loss(embedded_sets, sizes)

        return set_sizes_pred, size_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int, help='load a specific step from the latest run. -1 to load last '
                                                       'saved training step')
    parser.add_argument('-p', '--predictor', action='store_true', help='train the size predictor using the existing '
                                                                       'tspn model')

    args = parser.parse_args()

    config = set_config()
    dataset = MnistSet(config.train_split, config.pad_value, 20)
    tspn = TspnAutoencoder(args.step, config, dataset)

    if args.predictor:
        tspn.train_size_predictor()
    else:
        tspn.train_tspn()
