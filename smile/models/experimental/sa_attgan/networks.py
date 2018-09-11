from functools import partial

import tensorflow as tf

from smile.ops import sn_conv
from smile.ops import sn_dconv


def encoder(x, is_training, **hparams):
    #sn_conv_lrelu = partial(sn_conv)

    # TODO: scope test

    with tf.variable_scope("foo"):
        z0 = x
        z1 = sn_conv(z0, 64, 4, 2)
        z2 = sn_conv(z1, 128, 4, 2)

    with tf.variable_scope("foo", reuse=True):
        z0 = x
        z1 = sn_conv(z0, 64, 4, 2)
        z2 = sn_conv(z1, 128, 4, 2)

    # TODO: Assert that only two weight kernels exist at this point.

    # TODO: OOOORRRR we can just explicitly set the scope names.

    exit()

    # Net definition.
    z0 = x
    z1 = sn_conv(z0, 64, 4, 2)
    z2 = sn_conv(z1, 128, 4, 2)

    print(z1)
    print(z2)

    print("hej")

    """
     bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    # Net definition.
    z0 = img
    z1 = conv_bn_lrelu(z0, 64, 4, 2)
    z2 = conv_bn_lrelu(z1, 128, 4, 2)
    z3 = conv_bn_lrelu(z2, 256, 4, 2)
    z4 = conv_bn_lrelu(z3, 512, 4, 2)
    z5 = conv_bn_lrelu(z4, 1024, 4, 2)

    return [z1, z2, z3, z4, z5]

    """


def decoder():
    pass


def classifier_discriminator_shared():
    pass


def classifier_private():
    pass


def discriminator_private():
    pass
