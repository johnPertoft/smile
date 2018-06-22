from functools import partial

import tensorflow as tf


"""
# TODO: See what differs in slim vs layers by replacing parts of graph creation.
from functools import partial
import tensorflow.contrib.slim as slim
conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = None
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm
"""


def encoder(img, is_training, **hparams):

    def reflect_pad(x, p):
        return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "reflect")

    conv = partial(tf.layers.conv2d, activation=None, use_bias=False, padding="valid")

    def conv_in_lrelu(x, d, k, s):
        x = reflect_pad(x, k // 2)
        x = conv(x, d, k, s)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.leaky_relu(x)
        return x

    # TODO: BN or IN for encoding layers?

    def res_block(x, d):
        x_input = x

        x = reflect_pad(x, 1)
        x = conv(x, d, 3, 1)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)

        x = reflect_pad(x, 1)
        x = conv(x, d, 3, 1)
        x = tf.contrib.layers.instance_norm(x)

        return x + x_input

    # Net definition.
    z0 = img
    z1 = conv_in_lrelu(z0, 32, 7, 1)
    z2 = conv_in_lrelu(z1, 128, 3, 2)
    z3 = conv_in_lrelu(z2, 256, 3, 2)
    bottleneck = z3
    for _ in range(6):
        bottleneck = res_block(bottleneck, 256)
    z4 = bottleneck

    # TODO: return bottleneck layers as well?
    return [z1, z2, z3, z4]


def decoder(zs, attributes, is_training, **hparams):

    dconv = partial(tf.layers.conv2d_transpose, activation=None, use_bias=False, padding="same")

    def dconv_in_lrelu(x, d, k, s):
        x = dconv(x, d, k, s)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.leaky_relu(x)
        return x

    attributes = attributes[:, tf.newaxis, tf.newaxis, :]

    def tile_attributes_like(z):
        h, w = z.get_shape()[1:3]
        return tf.tile(attributes, (1, h, w, 1))

    # Net definition.
    z = zs[-1]
    net = tf.concat((z, tile_attributes_like(z)), axis=3)
    net = dconv_in_lrelu(net, 128, 3, 2)
    net = tf.concat((net, zs[-3]), axis=3)  # Long skip connection.
    net = tf.concat((net, tile_attributes_like(net)), axis=3)  # Inject attributes again.
    net = dconv_in_lrelu(net, 64, 2, 2)
    net = tf.layers.conv2d(net, 3, 7, 1, activation=None, padding="same")
    net = tf.nn.tanh(net)

    return net


def classifier_discriminator_shared(img, is_training, **hparams):

    conv = partial(tf.layers.conv2d, activation=None, use_bias=False, padding="same")

    def conv_in_lrelu(x, d, k, s):
        x = conv(x, d, k, s)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.leaky_relu(x)
        return x

    # TODO: Currently (almost) same as paper.

    # Net definition.
    net = img
    net = conv_in_lrelu(net, 64, 4, 2)
    net = conv_in_lrelu(net, 128, 4, 2)
    net = conv_in_lrelu(net, 256, 4, 2)
    net = conv_in_lrelu(net, 512, 4, 2)

    return net


def classifier_private(h, n_classes, is_training, **hparams):
    # TODO: Stronger classifier should help?
        # regularization

    # Net definition.
    net = h
    net = tf.layers.conv2d(net, 1024, 3, 2, padding="same")
    net = tf.nn.leaky_relu(net)
    net = tf.layers.conv2d(net, n_classes, 3, 2, padding="valid")
    net = tf.squeeze(net, axis=(1, 2))

    return net


def discriminator_private(h, is_training, **hparams):

    # Net definition.
    net = h
    net = tf.layers.conv2d(net, 512, 3, 2, padding="same")
    net = tf.nn.leaky_relu(net)
    net = tf.layers.conv2d(net, 1, 3, 1, padding="same")
    net = tf.reduce_mean(net, axis=(1, 2))  # Patch GAN. TODO: Maybe more patches?

    return net
