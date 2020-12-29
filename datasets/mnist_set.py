import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class MnistSet:
    def __init__(self, train_split, pad_value, min_pixel_brightness=0):
        self.tfds_name = 'mnist'
        self.min_pixel_brightness = min_pixel_brightness
        self.element_size = 2
        self.max_num_elements = 360
        self.pad_value = pad_value
        self.image_size = 28
        self.train_split = train_split

    def pixels_to_set(self, pixels, label):
        xy = tf.squeeze(pixels)
        pixel_indices = tf.where(tf.greater(xy, tf.constant(self.min_pixel_brightness, dtype=tf.uint8))) / 28
        size = tf.shape(pixel_indices)[0]
        paddings = [[0, self.max_num_elements - tf.shape(pixel_indices)[0]], [0, 0]]
        padded = tf.cast(tf.pad(pixel_indices, paddings, 'CONSTANT', self.pad_value), tf.float32)
        return xy, padded, size, label

    def get_full_dataset(self):
        tfds.load(self.tfds_name, split='train+test', shuffle_files=True)

    def get_train_set(self):
        split = [f'train[:{self.train_split}%]']
        ds = tfds.load('mnist', split=split, shuffle_files=True)[0]
        assert isinstance(ds, tf.data.Dataset)
        ds = ds.map(lambda row: self.pixels_to_set(row["image"], row["label"]))
        ds.filter(lambda xy, padded, size, label: size > 50)
        return ds

    def get_val_set(self):
        split = [f'train[{self.train_split}%:]']
        dataset = tfds.load('mnist', split=split)[0]
        assert isinstance(dataset, tf.data.Dataset)
        dataset = dataset.map(lambda row: self.pixels_to_set(row["image"], row["label"]))
        return dataset

    def get_test_set(self):
        split = ['test']
        dataset = tfds.load('mnist', split=split)[0]
        assert isinstance(dataset, tf.data.Dataset)
        dataset = dataset.map(lambda row: self.pixels_to_set(row["image"], row["label"]))
        return dataset


if __name__ == '__main__':
    train = MnistSet(80, -999, 50).get_train_set()

    for sample in train.take(-1):
        raw = sample[0].numpy()
        pixel = sample[1].numpy()
        size = sample[2].numpy()
        x = pixel[:, 1]
        y = pixel[:, 0]
        plt.axis([0, 1, 0, 1])
        plt.imshow(raw)
        plt.scatter(x, y)
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()
