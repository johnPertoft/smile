import tensorflow as tf


def encoder(img, is_training, **hparams):
    def conv_bn_lrelu(x, d, k, s):
        x = tf.layers.conv2d(x, filters=d, kernel_size=k, strides=s, padding="same")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        return x

    # Net definition.
    net = img
    net = conv_bn_lrelu(net, 64, 4, 2)
    net = conv_bn_lrelu(net, 128, 4, 2)
    net = conv_bn_lrelu(net, 256, 4, 2)
    net = conv_bn_lrelu(net, 512, 4, 2)
    net = conv_bn_lrelu(net, 1024, 4, 2)

    return net


def decoder(z, attributes, is_training, **hparams):
    def deconv_bn_relu(x, d, k, s):
        x = tf.layers.conv2d_transpose(x, filters=d, kernel_size=k, strides=s, padding="same")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        return x

    # TODO: Use attributes. Concat as feature maps?

    # Net definition.
    net = z
    net = deconv_bn_relu(net, 1024, 4, 2)
    net = deconv_bn_relu(net, 512, 4, 2)
    net = deconv_bn_relu(net, 256, 4, 2)
    net = deconv_bn_relu(net, 128, 4, 2)
    net = tf.layers.conv2d_transpose(net, filters=3, kernel_size=4, strides=2, padding="same")
    net = tf.nn.tanh(net)

    return net


def classifier_discriminator_shared(img, is_training, **hparams):
    def conv_ln_lrelu(x, d, k, s):
        x = tf.layers.conv2d(x, filters=d, kernel_size=k, strides=s)
        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.leaky_relu(x)
        return x

    # Net definition.
    net = img
    net = conv_ln_lrelu(net, 64, 4, 2)
    net = conv_ln_lrelu(net, 128, 4, 2)
    net = conv_ln_lrelu(net, 256, 4, 2)
    net = conv_ln_lrelu(net, 512, 4, 2)
    net = conv_ln_lrelu(net, 1024, 4, 2)

    return net


def classifier_private(h, n_classes, is_training, **hparams):
    net = h
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.contrib.layers.layer_norm(net)
    net = tf.nn.leaky_relu(net)
    net = tf.layers.dense(net, n_classes)  # Note: This function outputs the logits only.

    return net


def discriminator_private(h, is_training, **hparams):
    net = h
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.contrib.layers.layer_norm(net)
    net = tf.nn.leaky_relu(net)
    net = tf.layers.dense(net, 1)  # Note: This function should have linearly activated output for wgan loss.

    return net
