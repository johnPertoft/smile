import tensorflow as tf

from smile.unit.loss import vae_loss, gan_losses


def preprocess(x):
    h, w = x.shape[1:-1]
    x = x * 2 - 1
    x = tf.image.resize_images(x, [h - 2, w - 2])
    return x


def postprocess(x):
    h, w = x.shape[1:-1]
    x = tf.image.resize_images(x, [h + 2, w + 2])
    x = (x + 1) / 2
    return


class UNIT:
    def __init__(self,
                 A, B,
                 private_encoder_fn, shared_encoder_fn,
                 shared_decoder_fn, private_decoder_fn,
                 discriminator_fn,
                 **hparams):

        # TODO: refactor, makes sense that encoder_fn and encoder_shared_fn should be more closely tied
        # to ensure compatibility.

        # Encoder subgraphs.
        encoder_a_private = tf.make_template("encoder_a_private", private_encoder_fn)
        encoder_b_private = tf.make_template("encoder_b_private", private_encoder_fn)
        encoder_shared = tf.make_template("encoder_shared", shared_encoder_fn)
        encoder_a = lambda x: encoder_shared(encoder_a_private(x))
        encoder_b = lambda x: encoder_shared(encoder_b_private(x))

        # Decoder subgraphs.
        decoder_shared = tf.make_template("decoder_shared", shared_decoder_fn)
        decoder_a_private = tf.make_template("decoder_a_private", private_decoder_fn)
        decoder_b_private = tf.make_template("decoder_b_private", private_decoder_fn)
        decoder_a = lambda x: decoder_a_private(decoder_shared(x))
        decoder_b = lambda x: decoder_b_private(decoder_shared(x))

        # Discriminator subgraphs.
        discriminator_a = tf.make_template("discriminator_a", discriminator_fn)
        discriminator_b = tf.make_template("discriminator_b", discriminator_fn)

        # Encoder computed posterior means.
        z_mu_a = encoder_a(A)
        z_mu_b = encoder_b(B)

        # Reparametrization for sampling of z from posterior.
        z_a = z_mu_a + tf.random_normal(tf.shape(z_mu_a), stddev=1.0)
        z_b = z_mu_b + tf.random_normal(tf.shape(z_mu_b), stddev=1.0)

        # Translations.
        a_translated = decoder_a(z_b)
        b_translated = decoder_b(z_a)

        # Losses.
        vae_loss_a = vae_loss(A, decoder_a(z_a), z_a)
        vae_loss_b = vae_loss(B, decoder_b(z_b), z_b)
        disc_loss_a, gen_loss_a = gan_losses(discriminator_a(A), discriminator_a(a_translated))
        disc_loss_b, gen_loss_b = gan_losses(discriminator_b(B), discriminator_b(b_translated))
