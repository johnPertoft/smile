import tensorflow as tf


def test_sample_target_attributes(attributes: tf.Tensor) -> tf.Tensor:
    n_samples = tf.shape(attributes)[0]
    n_classes = attributes.shape[1].value

    attributes_repeated = tf.reshape(tf.tile(attributes, [1, n_classes]), [-1, n_classes])
    active_attribute_repeated = tf.tile(tf.eye(n_classes), [n_samples, 1])

    target_attributes = tf.logical_xor(
        tf.cast(active_attribute_repeated, tf.bool),
        tf.cast(attributes_repeated, tf.bool))
    target_attributes = tf.cast(target_attributes, tf.float32)

    # TODO: Include option to get combinations of attributes.
    # TODO: Should add some indication of what is what.

    # TODO: Potentially we could take a transform fn: x, attr -> x'
    # and then create the full tensor of translated samples.

    return target_attributes
