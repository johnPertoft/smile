import tensorflow as tf


_CELEB_A_SHAPE = (218, 178, 3)


def celeb_input_fn(tfrecords_paths, num_epochs=None, batch_size=64, data_augmentation=False):
    def parse_serialized(serialized_example):
        features = tf.parse_single_example(serialized_example, features={"img": tf.FixedLenFeature([], tf.string)})
        img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _CELEB_A_SHAPE)
        img = tf.cast(img, tf.float32)
        img = img / 255
        return img

    # TODO: Put this in a common module.
    def augment(img):

        def central_crop_and_resize(x, crop_fraction):
            H, W, _ = _CELEB_A_SHAPE
            h_crop = int(H * crop_fraction / 2.0)
            w_crop = int(W * crop_fraction / 2.0)
            x = x[h_crop:H-h_crop, w_crop:W-w_crop, :]
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

        dataset = tf.data.Dataset.from_tensors(img)
        for fn in fns:
            dataset = dataset.concatenate(tf.data.Dataset.from_tensors(fn(img)))

        return dataset

    ds = tf.data.TFRecordDataset(tfrecords_paths)
    ds = ds.map(parse_serialized)
    if augment:
        ds = ds.flat_map(augment)
    ds = ds.shuffle(1024)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size)

    return ds.make_one_shot_iterator().get_next()
