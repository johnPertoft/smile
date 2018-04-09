import tensorflow as tf


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

        a_reconstructed = decoder_a(z_a)
        b_reconstructed = decoder_b(z_b)

        # TODO: Losses depend on
        # vae_a, vae_b
        # gan_a, gan_b

        prior = tf.distributions.Normal(loc=tf.zeros_like(z_a), scale=tf.ones_like(z_a))

        kl_div_a = tf.distributions.kl_divergence(
            tf.distributions.Normal(loc=z_a, scale=tf.ones_like(z_a)),
            prior)
        kl_div_b = tf.distributions.kl_divergence(
            tf.distributions.Normal(loc=z_b, scale=tf.ones_like(z_b)),
            prior)