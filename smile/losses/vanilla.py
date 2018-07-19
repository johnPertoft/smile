from typing import Callable, Tuple

import tensorflow as tf


def gan_losses(x_real: tf.Tensor,
               x_fake: tf.Tensor,
               discriminator: Callable[[tf.Tensor], tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses as defined in the original "Generative Adversarial Networks" paper.
    Reference: https://arxiv.org/abs/1406.2661
    :param x_real: Tensor holding the real samples.
    :param x_fake: Tensor holding the fake samples.
    :param discriminator: Callable returning the discriminator network's (linear) output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for discriminator and generator.
    """

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real,
        labels=tf.ones_like(d_real)))

    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.zeros_like(d_fake)))

    d_loss = d_real_loss + d_fake_loss

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.ones_like(d_fake)))

    return d_loss, g_loss


def improved_gan_losses(D_real, D_fake):
    """Goodfellow's improved loss formulation."""
    pass
