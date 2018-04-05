import tensorflow as tf


def conv(x, size, stride, n_filters):
    return tf.layers.conv2d(
        x,
        kernel_size=(size, size),
        strides=(stride, stride),
        filters=n_filters,
        activation=None,

    )

"""
    def conv7_stride1_k(inputs, k):
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
"""