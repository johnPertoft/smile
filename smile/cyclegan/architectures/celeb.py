import tensorflow as tf


# TODO: rename these to reflect what they implement.
# TODO: Make compatible with multi-gpu


def paper_generator(X, is_training, **hparams):
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

    def res_block(inputs, k, activation):
        net = inputs

        # Layer 1.
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect")
        net = tf.layers.conv2d(
            net,
            kernel_size=(3, 3),
            strides=(1, 1),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="valid")
        net = tf.contrib.layers.instance_norm(net)
        net = activation(net)

        # Layer 2.
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "reflect")
        net = tf.layers.conv2d(
            net,
            kernel_size=(3, 3),
            strides=(1, 1),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="valid")
        net = tf.contrib.layers.instance_norm(net)

        return inputs + net

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

    activation = tf.nn.elu  #tf.nn.relu  # TODO: Choose from hparams instead.
    norm = tf.contrib.layers.instance_norm

    # Net definition.
    net = X
    net = activation(norm(conv7_stride1_k(net, 32)))
    net = activation(norm(conv3_stride2_k(net, 64)))
    net = activation(norm(conv3_stride2_k(net, 128)))
    for i in range(6):
        net = res_block(net, 128, activation)
    net = activation(norm(deconv3_stride2_k(net, 64)))
    net = activation(norm(deconv3_stride2_k(net, 32)))
    net = tf.nn.tanh(conv7_stride1_k(net, 3))

    return net


def paper_discriminator(X, is_training, **hparams):
    weight_initializer = tf.truncated_normal_initializer(stddev=0.02)

    def conv4_stride2_k(inputs, k):
        return tf.layers.conv2d(
            inputs,
            kernel_size=(4, 4),
            strides=(2, 2),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="same")

    activation = tf.nn.leaky_relu
    norm = tf.contrib.layers.instance_norm

    # Net definition.
    net = X
    net = activation(conv4_stride2_k(net, 64))
    net = activation(norm(conv4_stride2_k(net, 128)))
    net = activation(norm(conv4_stride2_k(net, 256)))
    net = activation(norm(conv4_stride2_k(net, 512)))
    net = tf.layers.conv2d(
        net,
        kernel_size=(4, 4),
        strides=(1, 1),
        filters=1,
        activation=None,
        padding="same")

    # Note: The discriminator returns a tensor T of shape (?, x, y, 1)
    # where each T_(i,j) corresponds to the discriminator's output for one
    # larger patch of the input image.
    # See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39

    return net


# TODO: Implement this properly.
def unet_generator(X, is_training):
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

    def deconv3_stride2_k(inputs, shortcuts, k):
        """3x3. 2 strided deconvolution (transposed convolution) with k filters."""
        upsampled = tf.layers.conv2d_transpose(
            inputs,
            kernel_size=(3, 3),
            strides=(2, 2),
            filters=k,
            activation=None,
            kernel_initializer=weight_initializer,
            use_bias=False,
            padding="same")

        return tf.concat((upsampled, shortcuts), axis=-1)

    lrelu = tf.nn.leaky_relu
    norm = tf.contrib.layers.instance_norm

    # Net definition.  # TODO: See paper code.
    c1 = lrelu(norm(conv7_stride1_k(X, 32)))
    c2 = lrelu(norm(conv3_stride2_k(c1, 64)))
    c3 = lrelu(norm(conv3_stride2_k(c2, 128)))
    c4 = lrelu(norm(conv3_stride2_k(c3, 256)))
    d1 = lrelu(norm(deconv3_stride2_k(c4, c3, 128)))
    d2 = lrelu(norm(deconv3_stride2_k(d1, c2, 64)))
    d3 = lrelu(norm(deconv3_stride2_k(d2, c1, 32)))
    net = lrelu(norm(tf.layers.conv2d_transpose(d3, kernel_size=(3, 3), strides=(2, 2), filters=16, padding="same")))
    net = tf.nn.tanh(norm(tf.layers.conv2d(net, kernel_size=(3, 3), strides=(2, 2), filters=3, padding="same")))
    return net


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

    def dense_block(inputs, n_layers):

        for _ in range(n_layers):
            pass

    def transition(inputs):
        pass
        # convolution + pooling/strides/upsampling?
        # or bn, relu, conv = "composite function" from paper

    activation = tf.nn.elu  # tf.nn.relu  # TODO: Choose from hparams instead.
    norm = tf.contrib.layers.instance_norm

    # TODO: Don't know if this makes sense. Just replaced resnet block with some densely connected
    # conv layers compared to paper_generator atm.
    # Maybe better to do denseblock->down/upscaling->denseblock etc

    # Net definition.
    net = X
    net = activation(norm(conv7_stride1_k(net, 16)))
    net = dense_block()
    net = transition_downsample()
    net = dense_block()
    net = transition_upsample()
    # something like this

    # Net definition.
    net = X
    net = activation(norm(conv7_stride1_k(net, 32)))
    net = activation(norm(conv3_stride2_k(net, 64)))  # downsample
    net = activation(norm(conv3_stride2_k(net, 128)))  # downsample
    for _ in range(6):
        net = dense_block(net)
    net = activation(norm(deconv3_stride2_k(net, 64)))  # upsample
    net = activation(norm(deconv3_stride2_k(net, 32)))  # upsample
    net = tf.nn.tanh(conv7_stride1_k(net, 3))

    return net


GENERATORS = {
    "paper": paper_generator,
    "unet": unet_generator,
    "densenet": densenet_generator
}

DISCRIMINATORS = {
    "paper": paper_discriminator
}