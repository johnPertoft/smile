import functools

import tensorflow as tf


# Note: Paper assumes 64x64 images. This implementation is adapted to 128x128.


def generator(x, is_training, **hparams):
    conv = functools.partial(
        tf.layers.conv2d,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
    dconv = functools.partial(
        tf.layers.conv2d_transpose,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
    norm = functools.partial(tf.layers.batch_normalization, training=is_training)
    lrelu = tf.nn.leaky_relu
    relu = tf.nn.relu

    # TODO: Official implementation has n_filters=100 in last encoding layer? Not in paper.
    # TODO: Official implementation uses padding=1 except last encoding layer.
    # TODO: Official implementation seems to use feature matching as well. Not mentioned in paper.

    net = x

    net = lrelu(conv(net, 64, 4, 2, use_bias=True))
    net = lrelu(norm(conv(net, 128, 4, 2)))
    net = lrelu(norm(conv(net, 256, 4, 2)))
    net = lrelu(norm(conv(net, 512, 4, 2)))
    net = lrelu(norm(conv(net, 1024, 4, 1, padding="valid")))

    net = relu(norm(dconv(net, 512, 4, 1, padding="valid")))
    net = relu(norm(dconv(net, 256, 4, 2)))
    net = relu(norm(dconv(net, 128, 4, 2)))
    net = relu(norm(dconv(net, 64, 4, 2)))
    net = dconv(net, 3, 4, 2, use_bias=True)
    net = tf.nn.tanh(net)

    return net


def discriminator(x, is_training, **hparams):
    conv = functools.partial(
        tf.layers.conv2d,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
    norm = functools.partial(tf.layers.batch_normalization, training=is_training)
    lrelu = tf.nn.leaky_relu

    net = x
    net = lrelu(conv(net, 64, 4, 2, use_bias=True))
    net = lrelu(norm(conv(net, 128, 4, 2)))
    net = lrelu(norm(conv(net, 256, 4, 2)))
    net = lrelu(norm(conv(net, 512, 4, 2)))
    net = conv(net, 1, 4, 1, use_bias=True)

    # TODO: Want scalar output?

    return net
