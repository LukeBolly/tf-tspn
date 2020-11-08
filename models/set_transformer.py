import tensorflow as tf
from .transformer import scaled_dot_product_attention


class STEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d, h):
        super(STEncoder, self).__init__()

        # Embedding part
        self.linear_1 = tf.keras.layers.Dense(d, activation='relu')

        self.enc_layers = [EncoderLayer(d, h, RFF(d))
                           for _ in range(num_layers)]

    def call(self, x):
        x = self.linear_1(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int, rff):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead = MultiHeadAttention(d, h)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.rff = rff

    def call(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int, rff):
        super(EncoderLayer, self).__init__()
        self.mab = MultiHeadAttentionBlock(d, h, rff)

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


class RFF(tf.keras.layers.Layer):
    """
    Row-wise FeedForward layers.
    """

    def __init__(self, d):
        super(RFF, self).__init__()

        self.linear_1 = tf.keras.layers.Dense(d, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(d, activation='relu')
        self.linear_3 = tf.keras.layers.Dense(d, activation='relu')

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.linear_3(self.linear_2(self.linear_1(x)))
