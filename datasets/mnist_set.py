import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pydevd


class MnistSet:
    def __init__(self, pad_value, min_pixel_brightness=0):
        self.tfds_name = 'mnist'
        self.min_pixel_brightness = min_pixel_brightness
        self.element_size = 2
        self.max_num_elements = 342
        self.pad_value = pad_value

    def pixels_to_set(self, pixels, label):
        xy = tf.squeeze(pixels)
        pixel_indices = tf.where(tf.greater(xy, tf.constant(self.min_pixel_brightness, dtype=tf.uint8)))
        size = tf.shape(pixel_indices)[0]
        paddings = [[0, self.max_num_elements - tf.shape(pixel_indices)[0]], [0, 0]]
        padded = tf.pad(pixel_indices, paddings, 'CONSTANT', self.pad_value)
        return xy, padded, size, label

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
    train, val, test = MnistSet(-999).get_train_val_test()
    train = train.batch(100)

    for sample in train.take(-1):
        raw = sample[0].numpy()
        pixel = sample[1].numpy()
        size = sample[2].numpy()
        x = pixel[:, 1]
        y = pixel[:, 0]
        plt.imshow(raw)
        plt.scatter(x, y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
