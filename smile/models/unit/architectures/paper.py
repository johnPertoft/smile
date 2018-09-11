import functools

import tensorflow as tf


# TODO: Change initializer? Not specified in paper.

def conv(x, d, k, s, use_bias=False, padding="same"):
    return tf.layers.conv2d(
        x,
        kernel_size=(k, k),
        strides=(s, s),
        filters=d,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        use_bias=use_bias,
        padding=padding)


def dconv(x, d, k, s, use_bias=False):
    return tf.layers.conv2d_transpose(
        x,
        kernel_size=(k, k),
        strides=(s, s),
        filters=d,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        use_bias=use_bias,
        padding="same")


def res_block(x, d, k, norm, activation):
    x_orig = x
    x = activation(norm(conv(x, d, k, 1)))
    x = norm(conv(x, x_orig.shape[-1], k, 1))
    return x + x_orig


def encoder_private(x, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm = lambda x: x  # TODO: No normalization mentioned in paper?

    net = x
    net = activation(norm(conv(net, 64, 7, 1)))
    net = activation(norm(conv(net, 128, 3, 2)))
    net = activation(norm(conv(net, 256, 3, 2)))
    net = res_block(net, 512, 3, norm, activation)
    net = res_block(net, 512, 3, norm, activation)
    net = res_block(net, 512, 3, norm, activation)

    return net


def encoder_shared(h, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm = lambda x: x

    net = h
    net = res_block(net, 512, 3, norm, activation)

    return net


def decoder_shared(z, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm = lambda x: x

    net = z
    net = res_block(net, 512, 3, norm, activation)

    return net


def decoder_private(h, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm = lambda x: x

    net = h
    net = res_block(net, 512, 3, norm, activation)
    net = res_block(net, 512, 3, norm, activation)
    net = res_block(net, 512, 3, norm, activation)
    net = norm(activation(dconv(net, 256, 3, 2)))
    net = norm(activation(dconv(net, 128, 3, 2)))
    net = tf.nn.tanh(dconv(net, 3, 1, 1))

    return net


def discriminator(x, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm = lambda x: x

    net = x
    net = norm(activation(conv(net, 64, 3, 2)))
    net = norm(activation(conv(net, 128, 3, 2)))
    net = norm(activation(conv(net, 256, 3, 2)))
    net = norm(activation(conv(net, 512, 3, 2)))
    net = norm(activation(conv(net, 1024, 3, 2)))
    net = conv(net, 1, 2, 1)

    return net
