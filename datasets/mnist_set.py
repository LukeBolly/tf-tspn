import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class MnistSet:
    def __init__(self, min_pixel_brightness=0):
        self.tfds_name = 'mnist'
        self.min_pixel_brightness = min_pixel_brightness
        self.element_size = 2

    def pixels_to_set(self, pixels, label):
        xy = tf.squeeze(pixels)
        pixel_indices = tf.where(tf.greater(xy, tf.constant(self.min_pixel_brightness, dtype=tf.uint8)))
        return xy, pixel_indices, label

    def get_full_dataset(self):
        tfds.load(self.tfds_name, split='train+test', shuffle_files=True)

    def get_train_val_test(self, train_split_percent=80):
        split = [f'train[:{train_split_percent}%]', f'train[{train_split_percent}%:]', 'test']
        datasets = tfds.load('mnist', split=split, shuffle_files=True)
        for i in range(len(datasets)):
            assert isinstance(datasets[i], tf.data.Dataset)
            datasets[i] = datasets[i].map(lambda row: self.pixels_to_set(row["image"], row["label"]))

        return datasets


if __name__ == '__main__':
    ds = MnistSet().get_train_val_test()
    for sample in ds[0].take(-1):
        raw = sample[0].numpy()
        pixel = sample[1].numpy()
        x = pixel[:, 1]
        y = pixel[:, 0]
        plt.imshow(raw)
        plt.scatter(x, y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
