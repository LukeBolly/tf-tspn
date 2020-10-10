import tensorflow as tf
import torch
K = tf.keras.backend

def tf_piecewise_linear(weight, n_pieces, sizes):
    """
        Piecewise linear function. Evaluates f at the ratios in sizes.
        This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
    """
    # share same sequence length within each sample, so copy weighht across batch dim
    weight = tf.expand_dims(weight, axis=0)     # [1,2,21]
    weight = tf.broadcast_to(weight, [sizes.shape[0], weight.shape[1], weight.shape[2]])    # [2,2,21]

    # linspace [0, 1] -> linspace [0, n_pieces]
    index = n_pieces * sizes    # [2,3]
    index = tf.expand_dims(index, axis=1)   # [2, 1, 3]
    index = tf.broadcast_to(index, [index.shape[0], weight.shape[1], index.shape[2]])   # [2,2,3]

    # points in the weight vector to the left and right
    idx = tf.cast(index, dtype=tf.int64)
    frac = index - tf.floor(index)

    left = torch_gather(weight, idx)
    right = tf.clip_by_value(torch_gather(weight, (idx + 1)), clip_value_min=weight.dtype.min, clip_value_max=n_pieces)
    # right = weight.gather(2, (idx + 1).clamp(max=n_pieces))

    # interpolate between left and right point
    return (1 - frac) * left + frac * right


def torch_gather(x, indices):
    # if pytorch gather indices are [[[0, 10, 20], [0, 10, 20]], [[0, 10, 20], [0, 10, 20]]]
    # tf nd_gather needs to be
    # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20],
    #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20]]

    gather_axes = []
    # create indices for each axis
    for axis in range(len(indices.shape) - 1):
        ax = K.arange(0, indices.shape[axis], dtype=tf.int64)

        # indices will need to need to be repeated along each higher order axis
        for i in range(axis + 1):
            num_repeats = int(indices.shape[axis:].num_elements() / indices.shape[axis + i])
            num_repeats = [num_repeats for j in range(ax.shape[0])]
            ax = tf.repeat(ax, num_repeats, axis=0)
            ax = tf.expand_dims(ax, 0)

        gather_axes.append(tf.reshape(ax, [indices.shape.num_elements()]))
    gather_axes.append(tf.reshape(indices, [indices.shape.num_elements()]))

    # combine the indices from each dimension and reshape the target to what pytorch would return
    gather_indices = tf.stack(gather_axes, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped


def pytorch_piecewise_linear(weight, n_pieces, sizes):
    """
        Piecewise linear function. Evaluates f at the ratios in sizes.
        This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
    """
    # share same sequence length within each sample, so copy weighht across batch dim
    weight = weight.unsqueeze(0)
    weight = weight.expand(sizes.size(0), weight.size(1), weight.size(2))

    # linspace [0, 1] -> linspace [0, n_pieces]
    index = n_pieces * sizes
    index = index.unsqueeze(1)
    index = index.expand(index.size(0), weight.size(1), index.size(2))

    # points in the weight vector to the left and right
    idx = index.long()
    frac = index.frac()
    left = weight.gather(2, idx)
    right = weight.gather(2, (idx + 1).clamp(max=n_pieces))

    # interpolate between left and right point
    return (1 - frac) * left + frac * right

if __name__ == '__main__':

    tensor = [[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]]

    indices = [[[1, 1], [0, 1]], [[1, 0], [0, 0]]]

    res = torch_gather(tf.constant(tensor, dtype=tf.float32), tf.constant(indices, dtype=tf.int64))

    n_pieces = 20
    sizes_array = [[0, 0.5, 1], [0, 0.5, 1]]

    pt_sizes = torch.tensor(sizes_array).float()
    pt_c1 = torch.arange(0, 1, 1.0 / (n_pieces + 1))
    pt_c2 = torch.flip(torch.arange(0, 0.5, 0.5 / (n_pieces + 1)), [0])
    pt_weight = torch.stack([pt_c1, pt_c2]).float()
    pt_res = pytorch_piecewise_linear(pt_weight, n_pieces, pt_sizes)

    tf_sizes = tf.constant(sizes_array, dtype=tf.float32)
    tf_c1 = K.arange(0, 1, 1.0 / (n_pieces + 1), dtype=tf.float32)
    tf_c2 = tf.reverse(K.arange(0, 0.5, 0.5 / (n_pieces + 1), dtype=tf.float32), [0])
    tf_weight = tf.stack([tf_c1, tf_c2])
    tf_res = tf_piecewise_linear(tf_weight, n_pieces, tf_sizes)

    print(pt_res)
    print(tf_res)
