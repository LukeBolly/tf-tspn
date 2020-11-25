import io
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def set_to_plot(raw, true, sampled, predicted):
    num_elements = true.shape[0]
    img_size = 2
    plots_per_sample = 4
    figure = plt.figure(figsize=(img_size * plots_per_sample, num_elements * img_size))
    plt.grid(False)
    plt.tight_layout()

    # image
    for i in range(num_elements):
        # mnist image
        if raw is not None:
            plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(raw[i])

        # truth set
        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 2)
        x = true[i, :, 1]
        y = true[i, :, 0]
        plt.scatter(x, y)
        plt.axis([0, 1, 1, 0])

        # sampled set
        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 3)
        x = sampled[i, :, 1]
        y = sampled[i, :, 0]
        plt.scatter(x, y)
        plt.axis([0, 1, 1, 0])

        # predicted set
        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 4)
        x = predicted[i, :, 1]
        y = predicted[i, :, 0]
        plt.scatter(x, y)
        plt.axis([0, 1, 1, 0])

    return figure
