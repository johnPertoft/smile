from typing import Callable
from typing import Tuple

import tensorflow as tf


def gan_losses(x_real: tf.Tensor,
               x_fake: tf.Tensor,
               discriminator: Callable[[tf.Tensor], tf.Tensor],
               **hparams) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in the original "Generative Adversarial Networks" paper.
    Reference: https://arxiv.org/abs/1406.2661
    :param x_real: Real samples tensor.
    :param x_fake: Fake samples tensor.
    :param discriminator: Callable returning the discriminator network's (linear) output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for discriminator and generator.
    """

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real), d_real) + \
             tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake), d_fake)

    g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_fake), d_fake)

    return d_loss, g_loss


def non_saturating_gan_losses(x_real: tf.Tensor,
                              x_fake: tf.Tensor,
                              discriminator: Callable[[tf.Tensor], tf.Tensor],
                              **hparams) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in the original "Generative Adversarial Networks" paper. The non saturating version.
    (This is usually the standard when people use the normal gan loss.)
    Reference: https://arxiv.org/abs/1406.2661
    :param x_real: Real samples tensor.
    :param x_fake: Fake samples tensor.
    :param discriminator: Callable returning the discriminator network's (linear) output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for discriminator and generator.
    """

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_real), d_real) + \
             tf.losses.sigmoid_cross_entropy(tf.zeros_like(d_fake), d_fake)

    def log(x):
        return tf.log(x + 1e-7)

    g_loss = -tf.reduce_mean(log(tf.sigmoid(d_fake)))

    return d_loss, g_loss
