import tensorflow as tf


class UNIT:
    def __init__(self, A, B, encoder_fn, generator_fn, discriminator_fn, **hparams):

        with tf.variable_scope("encoder_a"):
            pass

        with tf.variable_scope("encoder_b"):
            pass

        with tf.variable_scope("encoder_shared"):
            pass

        with tf.variable_scope("decoder_shared"):
            pass

        with tf.variable_scope("decoder_a"):
            pass

        with tf.variable_scope("decoder_b"):
            pass

        discriminator_a = tf.make_template(discriminator_fn)
        discriminator_b = tf.make_template(discriminator_fn)

        # TODO: encoders need to share some layers. Maybe we can have
            # encoder_fn
            # shared_encoder_fn
            # shared_generator_fn
            # generator_fn
            # discriminator_fn

