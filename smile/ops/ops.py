import tensorflow as tf


def reflect_pad(x, p):
    return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "reflect")


# TODO: Easily composable ops for all models.

# TODO: Add spectral normed layers here.


def conv():
    pass


def dconv():
    pass
