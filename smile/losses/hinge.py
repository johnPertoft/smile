from typing import Callable, Tuple

import tensorflow as tf


def hinge_losses(x_real: tf.Tensor,
                 x_fake: tf.Tensor,
                 discriminator: Callable[[tf.Tensor], tf.Tensor],
                 **hparams) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns losses based on hinge loss as used in "Margin Adaptation for Generative Adversarial Networks".
    Reference: https://arxiv.org/abs/1704.03817
    :param x_real: Real samples tensor.
    :param x_fake: Fake samples tensor.
    :param discriminator: Callable returning the discriminator network's output for a given tensor.
                          Assumes variable sharing is taken care of (e.g. by tf.make_template).
    :return: Loss scalars for discriminator and generator.
    """

    # TODO: Adaptiveness from paper?

    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_loss = tf.reduce_mean(tf.maximum(1.0 - d_real, 0.0)) + \
             tf.reduce_mean(tf.maximum(1.0 + d_fake, 0.0))

    g_loss = -tf.reduce_mean(d_fake)

    return d_loss, g_loss
