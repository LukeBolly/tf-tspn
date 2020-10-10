import tensorflow as tf
K = tf.keras.backend


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
