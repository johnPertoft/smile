import tensorflow as tf


class AttGAN:
    def __init__(self,
                 imgs,
                 attributes,
                 encoder_fn,
                 decoder_fn,
                 discriminator_fn,
                 **hparams):

        is_training = tf.placeholder_with_default(False, [])

        encoder = tf.make_template("encoder", encoder_fn)
        decoder = tf.make_template("decoder", decoder_fn)
        discriminator = tf.make_template("discriminator", discriminator_fn)

        # TODO: Take a fn for shared parts of discriminator and classifier instead and construct both parts here
        # instead?

    def train_step(self):
        pass

    def export(self):
        pass
