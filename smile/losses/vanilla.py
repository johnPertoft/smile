from typing import Callable, Tuple

import tensorflow as tf


def gan_losses(x_real: tf.Tensor,
               x_fake: tf.Tensor,
               discriminator: Callable[[tf.Tensor], tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
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


def improved_gan_losses(D_real, D_fake):
    """Goodfellow's improved loss formulation."""
    pass
