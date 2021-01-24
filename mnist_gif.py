import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
from datasets import mnist_set
from datasets.mnist_set import MnistSet
from mnist_train import TspnAutoencoder
import mnist_train


class AnimatedMnist:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self):
        self.mnist = MnistSet(80, -1, 20)
        self.num_points = self.mnist.max_num_elements

        self.tspn = TspnAutoencoder(143001, mnist_train.set_config(), self.mnist)

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(4, 4))

        self.batch_size = 10
        self.frames_per_digit = 20
        self.total_frames = int(self.batch_size * ((self.frames_per_digit * 1.5)))

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
                                          init_func=self.setup_plot, blit=True, save_count=self.total_frames)

    def initialise_dataset(self):
        for images, sets, sizes, labels in self.mnist.get_train_set().batch(self.batch_size).skip(29).take(1):
            self.sets = sets
            self.sizes = sizes

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.initialise_dataset()
        self.stream = self.data_stream()

        x, y = next(self.stream).T
        self.scat = self.ax.scatter(x, y, s=150, c='green')
        self.ax.axis([0, 1, 1, 0], )
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        val_prior_loss, val_model_loss, sampled_elements, pred_set = self.tspn.eval_tspn_step(self.sets, self.sizes)

        for i in range(sampled_elements.shape[0]):

            initial = sampled_elements[i][:self.sizes[i]]
            final = pred_set[i][:self.sizes[i]]

            for frame in range(self.frames_per_digit + 1):
                xy = (self.frames_per_digit - frame) / self.frames_per_digit * initial + frame / self.frames_per_digit * final
                yield np.c_[xy[:, 1], xy[:, 0]]

                if frame == self.frames_per_digit:
                    for j in range(int(self.frames_per_digit / 2)):
                        yield np.c_[xy[:, 1], xy[:, 0]]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def save(self):
        self.ani.save('animation.gif', writer='imagemagick', fps=30)


if __name__ == '__main__':
    a = AnimatedMnist()
    a.save()
    plt.show()