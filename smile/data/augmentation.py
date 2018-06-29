from functools import partial

import tensorflow as tf

"""
def central_crop_and_resize(x, crop_fraction):
    H, W, _ = _CELEB_A_SHAPE
    h_crop = int(H * crop_fraction / 2.0)
    w_crop = int(W * crop_fraction / 2.0)
    x = x[h_crop:H - h_crop, w_crop:W - w_crop, :]
    x = tf.image.resize_images(x, [H, W])
    return x
"""


def _data_augmentation(img, *args):

    fns = [
        tf.image.flip_left_right,
        partial(tf.image.random_brightness, max_delta=0.25),
        partial(tf.image.random_saturation, lower=0.5, upper=2.0)
    ]

    # TODO: Include *args

    ds = tf.data.Dataset.from_tensors(img)
    for fn in fns:
        ds = ds.concatenate(tf.data.Dataset.from_tensors(fn(img)))

    return ds
