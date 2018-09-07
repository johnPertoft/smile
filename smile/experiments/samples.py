from typing import Callable
from typing import Tuple

import tensorflow as tf


def _repeat_elements(x, n):
    # Assumes first dimension is batch dimension.
    element_shape = x.get_shape().as_list()[1:]
    return tf.reshape(tf.tile(x, [1, n] + [1] * (len(element_shape) - 1)), [-1] + element_shape)


def _make_target_attributes(attributes: tf.Tensor) -> Tuple[tf.Tensor, int]:
    n_samples = tf.shape(attributes)[0]
    n_classes = attributes.shape[1].value

    attributes_repeated = _repeat_elements(attributes, n_classes)
    active_attribute_repeated = tf.tile(tf.eye(n_classes), [n_samples, 1])

    target_attributes = tf.logical_xor(
        tf.cast(active_attribute_repeated, tf.bool),
        tf.cast(attributes_repeated, tf.bool))
    target_attributes = tf.cast(target_attributes, tf.float32)

    # TODO: Include option to get combinations of attributes.
    n_targets_per_sample = n_classes

    return target_attributes, n_targets_per_sample


def multi_attribute_translation_samples(x: tf.Tensor,
                                        attributes: tf.Tensor,
                                        translate_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> tf.Tensor:

    target_attributes, n_targets_per_sample = _make_target_attributes(attributes)
    x_repeated = _repeat_elements(x, n_targets_per_sample)
    x_translated = translate_fn(x_repeated, target_attributes)
    x_translated = tf.reshape(x_translated, [-1, n_targets_per_sample] + x.get_shape().as_list()[1:])
    x_translated = tf.transpose(x_translated, [1, 0, 2, 3, 4])

    x_translation_samples = tf.concat((
        tf.expand_dims(x, 0),
        x_translated),
        axis=0)

    return x_translation_samples
