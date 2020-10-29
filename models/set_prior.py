import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfpl = tfp.layers
tfd = tfp.distributions
tfkl = tf.keras.layers
tfb = tfp.bijectors


class SetPrior(tf.keras.Model):
    def __init__(self, max_size, event_size, *args, **kwargs):
        super(SetPrior, self).__init__()
        self.max_size = max_size
        self.event_size = event_size
        mvnd_input_size = 2         # size 2 because loc and scale inputs

        self.parametrization = tfpl.VariableLayer([self.event_size, mvnd_input_size],
                                                  name='loc')
        self.learnable_mvndiag = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., 0],
                scale_diag=tf.exp(t[..., 1])
            )
        )

    def call(self, batch_size):
        # doesnt matter what we pass in here as tf.VariableLayer ignores input (an error gets thrown if empty though)
        params = self.parametrization(None)
        tiled = tf.tile(tf.expand_dims(params, 0), [batch_size * self.max_size, 1, 1])
        samples = self.learnable_mvndiag(tiled)
        shaped = tf.reshape(samples, [batch_size, self.max_size, self.event_size])
        return shaped


if __name__ == '__main__':
    batch_size = 10
    max_size = 342
    event_size = 2
    # set_sizes = [109, 85, 73, 100, 124, 151]

    prior = SetPrior(max_size, event_size)
    sample = prior(batch_size)

    plt.scatter(sample[..., 0], sample[..., 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
