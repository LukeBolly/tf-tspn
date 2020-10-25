import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfpl = tfp.layers
tfd = tfp.distributions
tfkl = tf.keras.layers
tfb = tfp.bijectors


class SetPrior(tf.keras.Model):
    def __init__(self, event_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_size = event_size
        mvnd_input_size = 2         # size 2 because loc and scale inputs

        self.parametrization = tfpl.VariableLayer([mvnd_input_size, self.event_size], name='loc')
        self.learnable_mvndiag = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., 0],
                scale_diag=tf.exp(t[..., 1])
            )
        )

    def call(self, set_sizes):
        # doesnt matter what we pass in here as actual input is a tf.Variable
        params = self.parametrization(None)
        num_samples = tf.reduce_sum(set_sizes)
        tiled = tf.repeat(tf.expand_dims(params, 0), num_samples, 0)
        samples = self.learnable_mvndiag(tiled)
        shaped = tf.reshape(samples, [num_samples, self.event_size])
        return shaped


if __name__ == '__main__':
    event_size = 2
    set_sizes = [109, 85, 73, 100, 124, 151]

    prior = SetPrior(event_size)
    sample = prior(set_sizes)

    plt.scatter(sample[:, 0], sample[:, 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
