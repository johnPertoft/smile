from typing import Callable

import tensorflow as tf


def _repeat_elements(x, n):
    # Assumes first dimension is batch dimension.
    element_shape = x.get_shape().as_list()[1:]
    return tf.reshape(tf.tile(x, [1, n] + [1] * (len(element_shape) - 1)), [-1] + element_shape)


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

    # TODO: Make this fn private?

    # TODO: Use _repeat_elements

    return target_attributes


def attribute_translation_samples(x: tf.Tensor,
                                  attributes: tf.Tensor,
                                  translate_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> tf.Tensor:

    n_classes = attributes.shape[1].value

    target_attributes = test_sample_target_attributes(attributes)
    n_translations_per_sample = n_classes  # TODO: Subject to change if adding combinations.
    x_repeated = _repeat_elements(x, n_translations_per_sample)
    x_translated = translate_fn(x_repeated, target_attributes)
    x_translated = tf.reshape(x_translated, [-1, n_translations_per_sample] + x.get_shape().as_list()[1:])
    x_translated = tf.transpose(x_translated, [1, 0, 2, 3, 4])

    x_translation_samples = tf.concat((
        tf.expand_dims(x, 0),
        x_translated),
        axis=0)

    # TODO: Add indication of what is what somehow?

    return x_translation_samples
