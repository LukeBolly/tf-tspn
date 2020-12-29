import tensorflow as tf
import math


class SetTransformerEncoder(tf.keras.Model):
    def __init__(self, transformer_dim, num_heads, num_layers, output_dim):
        super(SetTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [SAB(transformer_dim, num_heads) for _ in range(num_layers)]
        self.out_projection = tf.keras.layers.Conv1D(output_dim, 1, kernel_initializer='glorot_uniform',
                                                  use_bias=True)

    def call(self, inputs, mask=None):
        x = inputs

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        x = self.out_projection(x)
        return x

class MAB(tf.keras.layers.Layer):
    def __init__(self, dim_v, num_heads):
        super(MAB, self).__init__()
        self.dim_V = dim_v
        self.num_heads = num_heads
        self.fc_q = tf.keras.layers.Dense(dim_v)
        self.fc_k = tf.keras.layers.Dense(dim_v)
        self.fc_v = tf.keras.layers.Dense(dim_v)
        self.ln0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc_o = tf.keras.layers.Dense(dim_v)

    def call(self, q, k, mask):
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)

        q_ = tf.concat(tf.split(q, self.num_heads, 2), 0)
        k_ = tf.concat(tf.split(k, self.num_heads, 2), 0)
        v_ = tf.concat(tf.split(v, self.num_heads, 2), 0)

        attn_logits = tf.matmul(q_, k_, transpose_b=True) / math.sqrt(self.dim_V)
        if mask is not None:
            # There is probably a better way to do this, but for simplicity just copy the transformations applied to
            # the data to ensure the mask is also consistent with it
            mask_ = tf.repeat(tf.expand_dims(mask, 2), self.num_heads, 2)
            mask_ = tf.concat(tf.split(mask_, self.num_heads, 2), 0)
            mask_ = tf.matmul(mask_, mask_, transpose_b=True)
            attn_logits += (mask_ * -1e9)

        a = tf.nn.softmax(attn_logits, 2)
        o = tf.concat(tf.split(q_ + tf.matmul(a, v_), self.num_heads, 0), 2)
        o = self.ln0(o)
        o = o + tf.nn.relu(self.fc_o(o))
        o = self.ln1(o)
        return o


class SAB(tf.keras.layers.Layer):
    def __init__(self, dim_out, num_heads):
        super(SAB, self).__init__()
        self.mab = MAB(dim_out, num_heads)

    def call(self, x, mask):
        return self.mab(x, x, mask)
