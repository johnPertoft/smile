import functools

import tensorflow as tf


def densenet_generator(X, is_training, **hparams):
    weight_initializer = tf.truncated_normal_initializer(stddev=0.02)

    def conv7_stride1_k(inputs, k):
        """7x7, 1 strided convolution with k filters."""
        padded = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "reflect")
        return tf.layers.conv2d(
            padded,
            kernel_size=(7, 7),
            strides=(1, 1),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="valid")

    def dense_block(x, n_layers, activation, norm):
        def conv(l):
            return tf.layers.conv2d(
                tf.pad(l, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect"),
                kernel_size=(3, 3),
                strides=(1, 1),
                filters=hparams["growth_rate"],
                use_bias=False,
                padding="valid")

        # TODO: bottleneck layers
        # Composite function.
        def H(l):
            l = norm(l)
            l = activation(l)
            l = conv(l)
            return l

        prev_layers = [x]
        for _ in range(n_layers):
            x = H(tf.concat(prev_layers, axis=3))
            prev_layers.append(x)

        return x

    def transition(x, sample_direction, n_filters):
        assert sample_direction in ("up", "down")
        conv = tf.layers.conv2d if sample_direction == "down" else tf.layers.conv2d_transpose
        # TODO: Paper has 1x1 conv + 2x2 pooling, keep the 1x1?
        # TODO: activation here as well
        return conv(
            x, #tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect"),  # TODO: dim issues
            kernel_size=(3, 3),
            strides=(2, 2),
            filters=n_filters,
            use_bias=False,
            padding="same")

    activation = tf.nn.elu  # tf.nn.relu
    norm = tf.contrib.layers.instance_norm

    dense_block = functools.partial(dense_block, activation=activation, norm=norm)

    # Net definition.
    net = X
    net = activation(norm(conv7_stride1_k(net, 16)))
    net = dense_block(net, 3)
    net = transition(net, "down", 64)
    net = dense_block(net, 3)
    net = transition(net, "down", 128)
    net = dense_block(net, 3)
    net = transition(net, "up", 128)
    net = dense_block(net, 3)
    net = transition(net, "up", 64)
    net = tf.nn.tanh(conv7_stride1_k(net, 3))

    return net


def densenet_generator2(X, is_training, **hparams):
    weight_initializer = tf.truncated_normal_initializer(stddev=0.02)

    """Same as paper generator but with resnet part replaced with one densenet block."""

    def conv7_stride1_k(inputs, k):
        """7x7, 1 strided convolution with k filters."""
        padded = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "reflect")
        return tf.layers.conv2d(
            padded,
            kernel_size=(7, 7),
            strides=(1, 1),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="valid")

    def conv3_stride2_k(inputs, k):
        """3x3, 2 strided convolution with k filters."""
        return tf.layers.conv2d(
            inputs,
            kernel_size=(3, 3),
            strides=(2, 2),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="same")

    def dense_block(x, n_layers, activation, norm):
        def conv(l):
            return tf.layers.conv2d(
                tf.pad(l, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect"),
                kernel_size=(3, 3),
                strides=(1, 1),
                filters=hparams["growth_rate"],
                use_bias=False,
                padding="valid")

        # TODO: bottleneck layers
        # Composite function.
        def H(l):
            l = norm(l)
            l = activation(l)
            l = conv(l)
            return l

        prev_layers = [x]
        for _ in range(n_layers):
            x = H(tf.concat(prev_layers, axis=3))
            prev_layers.append(x)

        return x

    def deconv3_stride2_k(inputs, k):
        """3x3. 2 strided deconvolution (transposed convolution) with k filters."""
        return tf.layers.conv2d_transpose(
            inputs,
            kernel_size=(3, 3),
            strides=(2, 2),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="same")

    activation = tf.nn.elu  # tf.nn.relu
    norm = tf.contrib.layers.instance_norm

    dense_block = functools.partial(dense_block, activation=activation, norm=norm)

    # Net definition.
    net = X
    net = activation(norm(conv7_stride1_k(net, 32)))
    net = activation(norm(conv3_stride2_k(net, 64)))
    net = activation(norm(conv3_stride2_k(net, 128)))
    net = dense_block(net, 10)
    net = activation(norm(deconv3_stride2_k(net, 64)))
    net = activation(norm(deconv3_stride2_k(net, 32)))
    net = tf.nn.tanh(conv7_stride1_k(net, 3))

    return net


# TODO: densenet discriminator
