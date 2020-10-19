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
        self.parametrization = tfpl.VariableLayer([event_size, event_size], name='loc')
        self.learnable_mvndiag = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., 0],
                scale_diag=tf.exp(t[..., 1])
            )
        )

    def call(self, num_samples):
        # doesnt matter what we pass in here as actual input is a tf.Variable
        params = self.parametrization(None)
        params = tf.tile(tf.expand_dims(params, 0), [num_samples, 1, 1])
        return self.learnable_mvndiag(params)


if __name__ == '__main__':
    prior = SetPrior(2)
    distribution = prior(100)
    samples = distribution.sample(2000).numpy()

    plt.scatter(samples[:, 0], samples[:, 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
