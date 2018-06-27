from typing import Callable, Tuple

import tensorflow as tf


def lsgan_losses(x_real: tf.Tensor,
                 x_fake: tf.Tensor,
                 discriminator: Callable[[tf.Tensor], tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in "Least Squares Generative Adversarial Networks."
    Reference: https://arxiv.org/abs/1611.04076
    :param x_real: Tensor holding the real samples.
    :param x_fake: Tensor holding the fake samples.
    :param discriminator: Callable returning the discriminator network's output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return:
    """

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_real_loss = tf.reduce_mean((d_real - 1.0) ** 2.0)
    d_fake_loss = tf.reduce_mean(d_fake ** 2.0)
    d_loss = (d_real_loss + d_fake_loss) / 2.0

    g_loss = tf.reduce_mean((d_fake - 1.0) ** 2.0)

    return d_loss, g_loss
