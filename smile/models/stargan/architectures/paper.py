import functools

import tensorflow as tf


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
    x = norm(conv(x, d, k, 1))
    return x + x_orig


def generator(x, attributes, is_training, **hparams):
    def concat_attributes(x, attributes):
        c = attributes[:, tf.newaxis, tf.newaxis, :]
        h, w = x.get_shape()[1:3]
        c = tf.tile(c, (1, h, w, 1))
        return tf.concat((x, c), axis=3)

    activation = tf.nn.relu
    norm = tf.contrib.layers.instance_norm

    # Net definition.
    net = concat_attributes(x, attributes)
    net = activation(norm(conv(net, 64, 7, 1)))
    net = activation(norm(conv(net, 128, 4, 2)))
    net = activation(norm(conv(net, 256, 4, 2)))
    for _ in range(6):
        res_block(net, 256, 3, norm, activation)
    net = activation(norm(dconv(net, 128, 4, 2)))
    net = activation(norm(dconv(net, 64, 4, 2)))
    net = tf.nn.tanh(conv(net, 3, 7, 1))

    return net


def classifier_discriminator_shared(x, is_training, **hparams):
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.01)

    net = x
    net = activation(conv(net, 64, 4, 2, use_bias=True))
    net = activation(conv(net, 128, 4, 2, use_bias=True))
    net = activation(conv(net, 256, 4, 2, use_bias=True))
    net = activation(conv(net, 512, 4, 2, use_bias=True))
    net = activation(conv(net, 1014, 4, 2, use_bias=True))
    net = activation(conv(net, 2048, 4, 2, use_bias=True))

    # TODO: Maybe remove some layers for 128x128?

    return net


def classifier_private(h, n_attributes, is_training, **hparams):
    k = h.get_shape()[1]
    logits = conv(h, n_attributes, k, 1, padding="valid")
    logits = tf.squeeze(logits, axis=[1, 2])
    return logits


def discriminator_private(h, is_training, **hparams):
    logits = conv(h, 1, 3, 1)
    return logits
