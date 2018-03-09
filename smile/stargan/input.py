import tensorflow as tf


_CELEB_A_SHAPE = (218, 178, 3)


def celeb_input_fn(tfrecords_paths, num_epochs=None, batch_size=64):
    """Return input tensors (img, label)."""

    def parse_serialized(serialized_example):
        features = tf.parse_single_example(serialized_example, features={"img": tf.FixedLenFeature([], tf.string)})
        img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _CELEB_A_SHAPE)
        img = tf.cast(img, tf.float32)
        img = img / 255
        return img  # TODO: Add the label too.

    return (tf.data.TFRecordDataset(tfrecords_paths)
            .map(parse_serialized)
            .shuffle(1024)
            .repeat(num_epochs)
            .batch(batch_size)
            .make_one_shot_iterator().get_next())
