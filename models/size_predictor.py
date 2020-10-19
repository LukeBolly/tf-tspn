import tensorflow as tf


class SizePredictor(tf.keras.Model):
    def __init__(self, hidden_size):
        super(SizePredictor).__init__()
        self.h1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.h2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        out1 = self.h1(inputs)
        out2 = self.h2(out1)
        return out2
