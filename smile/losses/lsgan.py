from typing import Callable
from typing import Tuple

import tensorflow as tf


def lsgan_losses(x_real: tf.Tensor,
                 x_fake: tf.Tensor,
                 discriminator: Callable[[tf.Tensor], tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in "Least Squares Generative Adversarial Networks."
    Reference: https://arxiv.org/abs/1611.04076
    :param x_real: Real samples tensor.
    :param x_fake: Fake samples tensor.
    :param discriminator: Callable returning the discriminator network's output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for discriminator and generator.
    """

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_loss = tf.losses.mean_squared_error(tf.ones_like(d_real), d_real) + \
             tf.losses.mean_squared_error(tf.zeros_like(d_fake), d_fake)
    d_loss = d_loss / 2.0

    g_loss = tf.losses.mean_squared_error(tf.ones_like(d_fake), d_fake)
    g_loss = g_loss / 2.0

    return d_loss, g_loss
