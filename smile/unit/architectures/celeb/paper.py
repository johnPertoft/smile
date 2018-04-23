import tensorflow as tf


def res_block(inputs, n_filters, activation):
    net = inputs

    # Layer 1.
    net = tf.layers.conv2d(
        tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect"),
        kernel_size=(3, 3),
        strides=(1, 1),
        filters=n_filters,
        activation=None,
        kernel_initializer=None,
        use_bias=False,
        padding="valid")
    net = tf.contrib.layers.instance_norm(net)
    net = activation(net)

    # Layer 2.
    net = tf.layers.conv2d(
        tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect"),
        kernel_size=(3, 3),
        strides=(1, 1),
        filters=n_filters,
        activation=None,
        kernel_initializer=None,
        use_bias=False,
        padding="valid")
    net = tf.contrib.layers.instance_norm(net)

    return inputs + net


def private_encoder(x, is_training, **hparams):

    def conv(inputs, k, s, n):
        ps = k // 2
        return tf.layers.conv2d(
            tf.pad(inputs, [[0, 0], [ps, ps], [ps, ps], [0, 0]], "reflect"),
            kernel_size=(k, k),
            strides=(s, s),
            filters=n,
            activation=None,
            kernel_initializer=None,
            use_bias=False,
            padding="valid")

    activation = tf.nn.leaky_relu
    # TODO: Normalization?

    # Net definition.
    net = x
    net = activation(conv(net, 7, 1, 64))
    net = activation(conv(net, 3, 2, 128))
    net = activation(conv(net, 3, 2, 256))
    for _ in range(3):
        net = res_block(net, 256, activation)

    return net


def shared_encoder(h, is_training, **hparams):
    activation = tf.nn.leaky_relu
    return res_block(h, 256, activation)


def shared_decoder(z, is_training, **hparams):
    activation = tf.nn.leaky_relu
    return res_block(z, 256, activation)


def private_decoder(h, is_training, **hparams):

    def deconv(inputs, k, s, n):
        return tf.layers.conv2d_transpose(
            inputs,
            kernel_size=(k, k),
            strides=(s, s),
            filters=n,
            activation=None,
            kernel_initializer=None,
            use_bias=False,
            padding="same")

    activation = tf.nn.leaky_relu

    # Net definition.
    net = h
    for _ in range(3):
        net = res_block(net, 256, activation)
    net = activation(deconv(net, 3, 2, 256))
    net = activation(deconv(net, 3, 2, 128))
    net = tf.nn.tanh(deconv(net, 1, 1, 3))

    return net


def discriminator(x, is_training, **hparams):
    def conv(inputs, k, s, n):
        return tf.layers.conv2d(
            inputs,
            kernel_size=(k, k),
            strides=(s, s),
            filters=n,
            activation=None,
            kernel_initializer=None,
            use_bias=False,
            padding="valid")

    activation = tf.nn.leaky_relu
    # TODO: Any normalization?

    # Net definition.
    net = x
    net = activation(conv(net, 3, 2, 64))
    net = activation(conv(net, 3, 2, 128))
    net = activation(conv(net, 3, 2, 256))
    net = activation(conv(net, 3, 2, 512))
    net = conv(net, 2, 1, 1)  # Note: sigmoid is added implicitly in loss.

    return net
