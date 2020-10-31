import tensorflow as tf

# function was not available in tfg-gpu at time of writing, this is an import from github
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py


def chamfer_distance(point_set_a, point_set_b, name=None):
    """Computes the Chamfer distance for the given two point sets.
    Note:
    This is a symmetric version of the Chamfer distance, calculated as the sum
    of the average minimum distance from point_set_a to point_set_b and vice
    versa.
    The average minimum distance from one point set to another is calculated as
    the average of the distances between the points in the first set and their
    closest point in the second set, and is thus not symmetrical.
    Note:
    This function returns the exact Chamfer distance and not an approximation.
    Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
    Args:
    point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
      represents points in a D dimensional space.
    point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
      represents points in a D dimensional space.
    name: A name for this op. Defaults to "chamfer_distance_evaluate".
    Returns:
    A tensor of shape `[A1, ..., An]` storing the chamfer distance between the
    two point sets.
    Raises:
    ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
    """
    with tf.compat.v1.name_scope(name, "chamfer_distance_evaluate", [point_set_a, point_set_b]):
        point_set_a = tf.convert_to_tensor(value=point_set_a)
        point_set_b = tf.convert_to_tensor(value=point_set_b)

        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
        # dimension D).
        difference = (
            tf.expand_dims(point_set_a, axis=-2) -
            tf.expand_dims(point_set_b, axis=-3))
        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(
            input_tensor=square_distances, axis=-1)
        minimum_square_distance_b_to_a = tf.reduce_min(
            input_tensor=square_distances, axis=-2)

        return (
            tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
