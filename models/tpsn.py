import tensorflow as tf
from models.fspool import FSEncoder
from models.transformer import Encoder
from models.set_prior import SetPrior


class Tspn(tf.keras.Model):
    def __init__(self, encoder_latent, encoder_out, fspool_n_pieces, transformer_layers, transformer_attn_size,
                 transformer_num_heads, num_element_features):

        self._prior = SetPrior(num_element_features)

        self._encoder = FSEncoder(encoder_latent, encoder_out, fspool_n_pieces)
        self._transformer = Encoder(transformer_layers, transformer_attn_size, transformer_num_heads,
                                    num_element_features)

        # initialise the output to predict points at the center of our canvas
        self._set_prediction = tf.keras.layers.Conv1D(num_element_features, 1, kernel_initializer='zeros',
                                                      bias_initializer=tf.keras.initializers.constant(0.5),
                                                      use_bias=True)

    def call(self, inputs, sizes):
        # encode the input set
        encoded = self._encoder(inputs, sizes)  # pooled: [batch_size, num_features]

        padded_samples = self.sample_prior(sizes)

        # concat the conditioning vector onto each element
        encoded_shaped = tf.tile(tf.expand_dims(encoded, 1), [1, self.max_set_size, 1])
        sampled_elements_conditioned = tf.concat([padded_samples, encoded_shaped], 2)

        masked_values = tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, self.max_set_size)), tf.float32)
        pred_set_latent = self._transformer(sampled_elements_conditioned, masked_values)

        pred_set = self._set_prediction(pred_set_latent)
        return pred_set

    def run_tspn(self, inputs, sizes):
        encoded = self._encoder(inputs, sizes)  # pooled: [batch_size, num_features]
        return encoded

    def run_prior(self, sizes):
        total_elements = tf.reduce_sum(sizes)
        sampled_elements = self._prior(total_elements)  # [batch_size, max_set_size, num_features]
        return sampled_elements

    def sample_prior(self, sizes):
        sampled_elements = self.run_prior(sizes)
        samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_elements, sizes)
        padded_samples = samples_ragged.to_tensor(default_value=self._c.pad_value,
                                                  shape=[sizes.shape[0], self.max_set_size, self.element_size])

        return padded_samples