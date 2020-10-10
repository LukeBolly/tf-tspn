import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def get_mnist_set_dataset(split=None, min_pixel_brightness=0):
    if split is None:
        split=['train[:80%]', 'train[80%:]', 'test']
    train_ds = tfds.load('mnist', split=split, shuffle_files=True)
    assert isinstance(train_ds, tf.data.Dataset)

    def pixels_to_set(pixels, label):
        xy = tf.squeeze(pixels)
        pixel_indices = tf.where(tf.greater(xy, tf.constant(min_pixel_brightness, dtype=tf.uint8)))
        return xy, pixel_indices, label

    set_dataset = train_ds.map(lambda x: pixels_to_set(x["image"], x["label"]))
    return set_dataset


if __name__ == '__main__':
    ds = get_mnist_set_dataset(split='train')
    for sample in ds.take(-1):
        raw = sample[0].numpy()
        set = sample[1].numpy()
        x = set[:, 1]
        y = set[:, 0]
        plt.imshow(raw)
        plt.scatter(x, y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
