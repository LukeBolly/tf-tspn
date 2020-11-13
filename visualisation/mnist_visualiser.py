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

def set_to_plot(raw, pixel, label):
    figure = plt.figure(figsize=(10, 100))
    for i in range(raw.shape[0]):
        # Start next subplot.
        plt.subplot(1, raw.shape[0], i + 1, title=str(label[i]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = pixel[i, :, 1]
        y = pixel[i, :, 0]
        plt.imshow(raw[0])
        plt.scatter(x, y)

    return figure