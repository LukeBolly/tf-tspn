import tensorflow as tf
from .layers.tf_torch_gather import torch_gather
K = tf.keras.backend


class FSPool(tf.keras.layers.Layer):
    """
        Featurewise sort pooling. From:
        FSPool: Learning Set Representations with Featurewise Sort Pooling.
    """

    def __init__(self, in_channels, n_pieces, relaxed=False):
        """
        in_channels: Number of channels in input
        n_pieces: Number of pieces in piecewise linear
        relaxed: Use sorting networks relaxation instead of traditional sorting
        """
        super().__init__()
        self.n_pieces = n_pieces
        self.relaxed = relaxed
        self.weight = self.add_weight("weight", shape=[in_channels, n_pieces + 1])

    def build(self, input_shape):
        pass

    def call(self, x, n=None):
        """ FSPool
        x: FloatTensor of shape (batch_size, in_channels, set size).
        This should contain the features of the elements in the set.
        Variable set sizes should be padded to the maximum set size in the batch with 0s.
        n: LongTensor of shape (batch_size).
        This tensor contains the sizes of each set in the batch.
        If not specified, assumes that every set has the same size of x.size(2).
        Note that n.max() should never be greater than x.size(2), i.e. the specified set size in the
        n tensor must not be greater than the number of elements stored in the x tensor.
        Returns: pooled input x, used permutation matrix perm
        """
        assert x.shape[1] == self.weight.shape[0], 'incorrect number of input channels in weight'
        # can call withtout length tensor, uses same length for all sets in the batch
        if n is None:
            n = tf.cast(tf.fill(x.shape[0], x.shape[2]), dtype=tf.int64)
        # create tensor of ratios $r$
        sizes, mask = fill_sizes(n, x)
        mask = tf.broadcast_to(mask, x.shape)

        # turn continuous into concrete weights
        weight = self.determine_weight(sizes)

        # make sure that fill value isn't affecting sort result
        # sort is descending, so put unreasonably low value in places to be masked away
        x = x + tf.cast((1 - mask), tf.float32) * -99999
        if self.relaxed:
            sorted_x, perm = cont_sort(x, temp=self.relaxed)
        else:
            sorted_x = tf.sort(x, axis=2, direction="DESCENDING")
            perm = tf.argsort(x, axis=2, direction="DESCENDING")

        x = tf.math.reduce_sum(sorted_x * weight * tf.cast(mask, tf.float32), 2)
        return x, perm


    def determine_weight(self, sizes):
        """
            Piecewise linear function. Evaluates f at the ratios in sizes.
            This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
        """
        # share same sequence length within each sample, so copy weighht across batch dim
        weight = tf.expand_dims(self.weight, axis=0)
        weight = tf.broadcast_to(weight, [sizes.shape[0], weight.shape[1], weight.shape[2]])  # [2,2,21]

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * sizes
        index = tf.expand_dims(index, axis=1)
        index = tf.broadcast_to(index, [index.shape[0], weight.shape[1], index.shape[2]])  # [2,2,3]

        # points in the weight vector to the left and right
        idx = tf.cast(index, dtype=tf.int64)
        frac = index - tf.floor(index)
        left = torch_gather(weight, idx, 2)
        right = torch_gather(weight,
                             tf.clip_by_value((idx + 1), clip_value_min=tf.int64.min, clip_value_max=self.n_pieces), 2)

        # interpolate between left and right point
        return (1 - frac) * left + frac * right


def fill_sizes(sizes, x=None):
    """
        sizes is a LongTensor of size [batch_size], containing the set sizes.
        Each set size n is turned into [0/(n-1), 1/(n-1), ..., (n-2)/(n-1), 1, 0, 0, ..., 0, 0].
        These are the ratios r at which f is evaluated at.
        The 0s at the end are there for padding to the largest n in the batch.
        If the input set x is passed in, it guarantees that the mask is the correct size even when sizes.max()
        is less than x.size(), which can be a case if there is at least one padding element in each set in the batch.
    """
    if x is not None:
        max_size = x.shape[2]
    else:
        max_size = tf.math.reduce_max(sizes)

    size_tensor = K.arange(start=0, stop=max_size, dtype=tf.float32)

    expanded = tf.expand_dims(size_tensor, axis=0)
    total_sizes = tf.expand_dims(tf.clip_by_value((tf.cast(sizes, dtype=tf.float32) - 1),
                                                  clip_value_min=1, clip_value_max=tf.float32.max), axis=1)
    size_tensor = tf.clip_by_value(tf.math.divide(expanded, total_sizes),
                                   clip_value_min=tf.float32.min, clip_value_max=1)

    mask = size_tensor <= 1
    mask = tf.cast(tf.expand_dims(mask, axis=1), dtype=tf.float32)

    return size_tensor, mask


def deterministic_sort(s, tau):
    """
    "Stochastic Optimization of Sorting Networks via Continuous Relaxations" https://openreview.net/forum?id=H1eSS3CcKX
    Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar.
    """
    n = s.shape[1]
    one = tf.ones((n, 1), dtype=tf.float32)
    A_s = tf.abs(s - tf.transpose(s, [0, 2, 1]))
    B = tf.linalg.matmul(A_s, tf.linalg.matmul(one, tf.transpose(one, [0, 1])))
    scaling = tf.cast(n + 1 - 2 * (K.arange(n) + 1), tf.float32)
    C = tf.linalg.matmul(s, tf.expand_dims(scaling, 0))
    P_max = tf.transpose(tf.subtract(C, B), [0, 2, 1])
    P_hat = tf.math.divide(P_max, tau)
    return tf.nn.softmax(P_hat)


def cont_sort(x, perm=None, temp=1):
    """ Helper function that calls deterministic_sort with the right shape.
    Since it assumes a shape of (batch_size, n, 1) while the input x is of shape (batch_size, channels, n),
    we can get this to the right shape by merging the first two dimensions.
    If an existing perm is passed in, we compute the "inverse" (transpose of perm) and just use that to unsort x.
    """
    original_size = x.shape
    x = tf.reshape(x, [-1, x.shape[2], 1])
    if perm is None:
        perm = deterministic_sort(x, temp)
    else:
        perm = tf.transpose(perm, [1, 2])
    x = tf.linalg.matmul(x)
    x = tf.reshape(x, original_size)
    return x, perm
