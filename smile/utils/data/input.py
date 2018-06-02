import tensorflow as tf


_CELEB_A_SHAPE = (218, 178, 3)

# TODO: Common input_fns
# One assuming each tf record example just contains "img" as raw byte encoded in [0, 255].
# One assuming each tf record example contains "img" as above and VarLenFeature of active attributes (strings).


def _data_augmentation(img, *args):
    def central_crop_and_resize(x, crop_fraction):
        H, W, _ = _CELEB_A_SHAPE
        h_crop = int(H * crop_fraction / 2.0)
        w_crop = int(W * crop_fraction / 2.0)
        x = x[h_crop:H - h_crop, w_crop:W - w_crop, :]
        x = tf.image.resize_images(x, [H, W])
        return x

    fns = [
        tf.image.flip_left_right,
        lambda x: tf.image.random_brightness(x, max_delta=0.25),
        lambda x: tf.image.random_saturation(x, lower=0.5, upper=2.0),
        lambda x: central_crop_and_resize(x, crop_fraction=0.25)
    ]

    # TODO: Add grayscale?
    # TODO: lambda x: tf.image.random_contrast(x, lower=0.01, 0.2), Good values?
    # TODO: args must be included but unchanged

    ds = tf.data.Dataset.from_tensors(img)
    for fn in fns:
        ds = ds.concatenate(tf.data.Dataset.from_tensors(fn(img)))

    return ds


def input_fn(tfrecord_paths, batch_size, num_epochs=None):

    def parse_serialized(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={"img": tf.FixedLenFeature([], tf.string)})
        img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _CELEB_A_SHAPE)
        img = tf.cast(img, tf.float32)
        img = img / 255
        return img

    # TODO: prefetch(_to_device)?
    # TODO: data augmentation

    ds = tf.data.TFRecordDataset(tfrecord_paths)
    ds = ds.map(parse_serialized)
    ds = ds.shuffle(1024)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)

    return ds


def input_fn_with_attributes(tfrecord_paths, considered_attributes, batch_size, num_epochs=None):

    # Note: One oov bucket for all non considered attributes.
    attribute_index = tf.contrib.lookup.index_table_from_tensor(considered_attributes, num_oov_buckets=1)

    def parse_serialized(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={"img": tf.FixedLenFeature([], tf.string),
                      "attributes": tf.VarLenFeature(tf.string)})

        # Images in [0, 1].
        img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _CELEB_A_SHAPE)
        img = tf.cast(img, tf.float32)
        img = img / 255

        # Considered attributes as indicator vector.
        attributes = attribute_index.lookup(features["attributes"])
        attributes = tf.sparse_to_indicator(attributes, len(considered_attributes) + 1)  # +1 for the oov bucket.
        attributes = attributes[:-1]  # Skip the oov bucket since those attributes should not be considered.
        attributes = tf.cast(attributes, tf.float32)

        return img, attributes

    def at_least_one_considered_attribute(unused_img, attributes):
        return tf.logical_not(tf.reduce_all(tf.equal(attributes, 0.0)))

    ds = tf.data.TFRecordDataset(tfrecord_paths)
    ds = ds.map(parse_serialized)
    ds = ds.filter(at_least_one_considered_attribute)  # TODO: Should be optional?
    ds = ds.shuffle(1024)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)

    # TODO: Can we avoid requiring an initializable iterator?

    return ds
