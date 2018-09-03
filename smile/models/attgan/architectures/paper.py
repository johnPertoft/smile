import tensorflow as tf


# TODO: What is described in the papers differs from their implementation.
    # Instance norm instead of layer norm (makes sense)
    # shortcut layers are used
    # inject layers are used
    # + other small differences compared to described architecture.


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


# TODO: Clean up this implementation. Ignore discrepancies between paper and implementation?

def encoder(img, is_training, **hparams):
    #def conv_bn_lrelu(x, d, k, s):
    #    x = tf.layers.conv2d(x, filters=d, kernel_size=k, strides=s, use_bias=False, padding="same")
    #    x = tf.layers.batch_normalization(x, training=is_training)
    #    x = tf.nn.leaky_relu(x)
    #    return x

    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    # Net definition.
    z0 = img
    z1 = conv_bn_lrelu(z0, 64, 4, 2)
    z2 = conv_bn_lrelu(z1, 128, 4, 2)
    z3 = conv_bn_lrelu(z2, 256, 4, 2)
    z4 = conv_bn_lrelu(z3, 512, 4, 2)
    z5 = conv_bn_lrelu(z4, 1024, 4, 2)

    return [z1, z2, z3, z4, z5]


def decoder(zs, attributes, is_training, **hparams):
    #def deconv_bn_relu(x, d, k, s):
    #    x = tf.layers.conv2d_transpose(x, filters=d, kernel_size=k, strides=s, use_bias=False, padding="same")
    #    x = tf.layers.batch_normalization(x, training=is_training)
    #    x = tf.nn.relu(x)
    #    return x

    bn = partial(batch_norm, is_training=is_training)
    deconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    attributes = attributes[:, tf.newaxis, tf.newaxis, :]

    def tile_attributes_like(z):
        h, w = z.get_shape()[1:3]
        return tf.tile(attributes, (1, h, w, 1))

    # TODO: Inject layers?
    # TODO: Shortcut layers? Maybe put this in another file as it wasnt actually mentioned in paper.

    # TODO: Definable by hparams.
    # Net definition.
    z = zs[-1]
    net = tf.concat((z, tile_attributes_like(z)), axis=3)
    net = deconv_bn_relu(net, 1024, 4, 2)
    net = tf.concat((net, zs[-2]), axis=3)
    net = tf.concat((net, tile_attributes_like(zs[-2])), axis=3)
    net = deconv_bn_relu(net, 512, 4, 2)
    #net = tf.concat((net, zs[-3]), axis=3)
    net = deconv_bn_relu(net, 256, 4, 2)
    net = deconv_bn_relu(net, 128, 4, 2)
    net = tf.layers.conv2d_transpose(net, filters=3, kernel_size=4, strides=2, padding="same")
    net = tf.nn.tanh(net)

    return net


def classifier_discriminator_shared(img, is_training, **hparams):
    #def conv_in_lrelu(x, d, k, s):
    #    x = tf.layers.conv2d(x, filters=d, kernel_size=k, strides=s)
    #    x = tf.contrib.layers.instance_norm(x)
    #    x = tf.nn.leaky_relu(x)
    #    return x

    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    # Net definition.
    net = img
    net = conv_in_lrelu(net, 64, 4, 2)
    net = conv_in_lrelu(net, 128, 4, 2)
    net = conv_in_lrelu(net, 256, 4, 2)
    net = conv_in_lrelu(net, 512, 4, 2)
    net = conv_in_lrelu(net, 1024, 4, 2)

    return net


def classifier_private(h, n_classes, is_training, **hparams):
    net = h
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    #net = tf.contrib.layers.instance_norm(net)  # TODO: paper mentions this but not on github.
    net = tf.nn.leaky_relu(net)
    net = tf.layers.dense(net, n_classes)  # Note: This function outputs the logits only.

    return net


def discriminator_private(h, is_training, **hparams):
    net = h
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    #net = tf.contrib.layers.instance_norm(net)
    net = tf.nn.leaky_relu(net)
    net = tf.layers.dense(net, 1)  # Note: This function should have linearly activated output for wgan loss.

    return net
