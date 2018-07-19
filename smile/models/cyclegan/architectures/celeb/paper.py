import functools

import tensorflow as tf

from smile.utils.ops import reflect_pad


weight_initializer = tf.truncated_normal_initializer(stddev=0.02)


def conv(x, d, k, s, norm_fn=None, activation_fn=None):
    x = reflect_pad(x, k // 2)
    x = tf.layers.conv2d(
        x,
        filters=d,
        kernel_size=k,
        strides=s,
        activation=None,
        kernel_initializer=weight_initializer,
        use_bias=norm_fn == tf.contrib.layers.instance_norm,
        padding="valid")
    if norm_fn is not None:
        x = norm_fn(x)
    if activation_fn is not None:
        x = activation_fn(x)
    return x


def dconv(x, d, k, s, norm_fn=None, activation_fn=None):
    #x = reflect_pad(x, k // 2)  # TODO: reflect padding here?
    x = tf.layers.conv2d_transpose(
        x,
        filters=d,
        kernel_size=k,
        strides=s,
        activation=None,
        kernel_initializer=weight_initializer,
        use_bias=norm_fn == tf.contrib.layers.instance_norm,
        padding="same")
    if norm_fn is not None:
        x = norm_fn(x)
    if activation_fn is not None:
        x = activation_fn(x)
    return x


def generator(X, is_training, **hparams):
    # TODO: paper/implementation difference.
    # Paper just states that reflection padding is used but pytorch implementation uses zero padding
    # in up- and down-sample layers.

    norm_fn = tf.contrib.layers.instance_norm  # TODO: pytorch code sets affine=False. trainable=False is equiv?
    activation_fn = tf.nn.relu

    conv_norm_activation = functools.partial(
        conv,
        norm_fn=norm_fn,
        activation_fn=activation_fn)

    dconv_norm_activation = functools.partial(
        dconv,
        norm_fn=norm_fn,
        activation_fn=activation_fn)

    def res_block(x, d):
        x_orig = x

        x = conv(x, d, 3, 1)
        x = norm_fn(x)
        x = activation_fn(x)

        x = conv(x, d, 3, 1)
        x = norm_fn(x)

        return x + x_orig

    # Net definition.
    net = X
    net = conv_norm_activation(net, 32, 7, 1)
    net = conv_norm_activation(net, 64, 3, 2)
    net = conv_norm_activation(net, 128, 3, 2)
    for _ in range(6):
        net = res_block(net, 128)
    net = dconv_norm_activation(net, 64, 3, 2)
    net = dconv_norm_activation(net, 32, 3, 2)
    net = conv(net, 3, 7, 1)
    net = tf.nn.tanh(net)

    return net


def discriminator(X, is_training, **hparams):
    norm_fn = tf.contrib.layers.instance_norm
    activation_fn = tf.nn.leaky_relu

    conv_norm_activation = functools.partial(
        conv,
        norm_fn=norm_fn,
        activation_fn=activation_fn)

    # Net definition.
    net = X
    net = activation_fn(conv(net, 64, 4, 2))
    net = conv_norm_activation(net, 128, 4, 2)
    net = conv_norm_activation(net, 256, 4, 2)
    net = conv_norm_activation(net, 512, 4, 2)
    net = conv(net, 1, 4, 1)

    # Note: This is patch-gan, i.e. the output is a tensor of shape (?, h, w, 1)
    # where each element corresponds to the discriminator's output for a larger input
    # patch.

    return net
