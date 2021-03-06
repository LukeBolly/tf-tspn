import tensorflow as tf
from datasets.mnist_set import MnistSet


# function was not available in tfg-gpu at time of writing, this is an import from github
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py

# modification of standard chamfer distance, using huber loss instead of squared distance
# same loss used in DSPN, not sure about TSPN
def chamfer_distance_smoothed(point_set_a, point_set_b, sizes, name=None):
    with tf.compat.v1.name_scope(name, "chamfer_distance_evaluate", [point_set_a, point_set_b]):
        point_set_a = tf.convert_to_tensor(value=point_set_a)
        point_set_b = tf.convert_to_tensor(value=point_set_b)

        a = tf.expand_dims(point_set_a, axis=-2)
        b = tf.expand_dims(point_set_b, axis=-3)

        square_distances = tf.keras.losses.huber(a, b)

        # remove the padded values before finding the min distance, otherwise the model can abuse the padding to
        # achieve lower chamfer loss and not actually learn anything
        # slice off the known extras from our tensor, otherwise raggedTensor throws an error if the final ragged
        # tensor can be squeezed smaller than the initial size (ie. at least one row / column needs to be current size)
        largest_unpadded_dim = tf.reduce_max(sizes)
        square_distances = square_distances[:, :largest_unpadded_dim, :largest_unpadded_dim]

        row_sizes = tf.repeat(sizes, sizes)
        square_distances = tf.RaggedTensor.from_tensor(square_distances, lengths=(sizes, row_sizes))

        minimum_square_distance_a_to_b = tf.reduce_min(input_tensor=square_distances, axis=-1)
        minimum_square_distance_b_to_a = tf.reduce_min(input_tensor=square_distances, axis=-2)

        setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
        return setwise_distance


if __name__ == '__main__':
    set_a = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8]], [[2], [3], [4], [5], [6], [7], [8], [9]]])
    set_b = tf.constant([[[8], [7], [6], [5], [4], [3], [2], [1]], [[8], [7], [6], [5], [4], [3], [2], [1]]])
    # min distances        3    2    1    0    0                           1    0    0    0    0    0
    # mean                 6 / 5 = 5.6                                     1 / 6 = 0.333
    dist = chamfer_distance_smoothed(set_a, set_b, [5, 6])
    print(dist.numpy())

    train = MnistSet(80, -999).get_train_set()
    train = train.batch(10)

    for sample in train.take(-1):
        raw = sample[0].numpy()
        pixel = sample[1].numpy()
        size = sample[2].numpy()

        # distance of two same sets should be 0
        dist = chamfer_distance_smoothed(pixel, pixel, size)
        print(dist.numpy())
