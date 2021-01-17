import tensorflow as tf


class SizePredictor(tf.keras.Model):
    def __init__(self, hidden_size, max_units):
        super(SizePredictor, self).__init__()
        self.h1 = tf.keras.layers.Dense(hidden_size, kernel_initializer='glorot_uniform')
        self.h2 = tf.keras.layers.Dense(max_units, kernel_initializer='glorot_uniform')

    def call(self, inputs, training=None, mask=None):
        out1 = self.h1(inputs)
        out1 = tf.nn.relu(out1)
        out2 = self.h2(out1)
        return out2
