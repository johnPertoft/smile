import tensorflow as tf


def conv():
    # TODO: Mostly act as a pass through to normal tf conv
    # TODO: Add spectral norm options
    pass


def dconv():
    pass


def res_block():
    pass


"""

def sn_conv(x, d, k, s, use_bias=True, padding="SAME", n_power_iterations=1):
    # TODO: Add assertion about data format. NHWC is expected.

    with tf.variable_scope(None, default_name="sn_conv"):
        d_in = x.shape[-1]
        d_out = d

        kernel = tf.get_variable("kernel", shape=(k, k, d_in, d_out), initializer=tf.initializers.variance_scaling())
        kernel = spectral_normalization(kernel, n_power_iterations)

        x = tf.nn.conv2d(x, kernel, strides=(1, s, s, 1), padding=padding)
        if use_bias:
            b = tf.get_variable("b", shape=(d_out,), initializer=tf.initializers.constant(0.0))
            x = tf.nn.bias_add(x, b)

        return x


def sn_dconv(x, d, k, s, use_bias=True, padding="SAME", n_power_iterations=1):
    # TODO: Add assertion about data format. NHWC is expected.
    # TODO: Is there any concern for the different order of dimensions for the deconv kernel?

    with tf.variable_scope(None, default_name="sn_dconv"):
        d_in = x.shape[-1]
        d_out = d

        kernel = tf.get_variable("kernel", shape=(k, k, d_out, d_in), initializer=tf.initializers.variance_scaling())
        kernel = spectral_normalization(kernel, n_power_iterations)

        x = tf.nn.conv2d_transpose(x, kernel, strides=(1, s, s, 1), padding=padding)
        if use_bias:
            b = tf.get_variable("b", shape=(d_out,), initializer=tf.initializers.constant(0.0))
            x = tf.nn.bias_add(x, b)
        return x
"""
