from typing import Callable, Tuple

import tensorflow as tf


def wgan_gp_losses(x_real: tf.Tensor,
                   x_fake: tf.Tensor,
                   critic: Callable[[tf.Tensor], tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in "Improved Training of Wasserstein GANs".
    Reference: https://arxiv.org/abs/1704.00028
    :param x_real: Tensor holding the real samples.
    :param x_fake: Tensor holding the fake samples.
    :param critic: Callable returning the critic network's output for a given tensor.
                   Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: The critic and generator loss.
    """

    # Interpolate between x_real and x_fake.
    shape = [tf.shape(x_real)[0]] + [1] * (x_real.shape.ndims - 1)
    epsilon = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    x_interpolate = epsilon * x_real + (1.0 - epsilon) * x_fake

    # Gradient penalty.
    predictions = critic(x_interpolate)
    gradients = tf.gradients(predictions, x_interpolate)[0]
    norm = tf.norm(tf.layers.flatten(gradients), axis=1)
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2.0)  # TODO: Try with l1 norm as well? or make it optional

    # Losses for critic and generator.
    c_real = critic(x_real)
    c_fake = critic(x_fake)
    critic_loss = -tf.reduce_mean(c_real) + tf.reduce_mean(c_fake) + gradient_penalty * 10.0
    generator_loss = -tf.reduce_mean(c_fake)

    return critic_loss, generator_loss
