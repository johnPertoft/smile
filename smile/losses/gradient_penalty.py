from typing import Callable

import tensorflow as tf


def gradient_penalty(x_hat: tf.Tensor,
                     discriminator: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:

    d_hat = discriminator(x_hat)
    gradients = tf.gradients(d_hat, x_hat)[0]
    norm = tf.norm(tf.layers.flatten(gradients), axis=1)
    gp = tf.reduce_mean((norm - 1.0) ** 2.0)

    # TODO: Option for other norms as well?

    return gp
