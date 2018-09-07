from typing import Callable
from typing import Tuple

import tensorflow as tf

from .gradient_penalty import gradient_penalty


def wgan_gp_losses(x_real: tf.Tensor,
                   x_fake: tf.Tensor,
                   critic: Callable[[tf.Tensor], tf.Tensor],
                   **hparams) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in "Improved Training of Wasserstein GANs".
    Reference: https://arxiv.org/abs/1704.00028
    :param x_real: Real samples tensor.
    :param x_fake: Fake samples tensor.
    :param critic: Callable returning the critic network's output for a given tensor.
                   Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for critic and generator.
    """

    # Interpolate between x_real and x_fake.
    shape = [tf.shape(x_real)[0]] + [1] * (x_real.shape.ndims - 1)
    epsilon = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    x_interpolate = epsilon * x_real + (1.0 - epsilon) * x_fake

    gp = gradient_penalty(x_interpolate, critic)

    # Losses for critic and generator.
    c_real = critic(x_real)
    c_fake = critic(x_fake)
    critic_loss = -tf.reduce_mean(c_real) + tf.reduce_mean(c_fake) + gp * hparams["wgan_gp_lambda"]
    generator_loss = -tf.reduce_mean(c_fake)

    return critic_loss, generator_loss
