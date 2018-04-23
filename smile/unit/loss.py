import tensorflow as tf


def kl_divergence(z_mu):
    # Hard coded kl divergence with unit normal.
    prior = tf.distributions.Normal(loc=tf.zeros_like(z_mu), scale=tf.ones_like(z_mu))
    q_z_x = tf.distributions.Normal(loc=z_mu, scale=tf.ones_like(z_mu))
    kl_div = tf.distributions.kl_divergence(q_z_x, prior)
    return tf.reduce_mean(kl_div)


def nll(x, x_rec):
    # A.k.a. reconstruction loss. Assumes decoder p(x|z) are modeled as Laplacian distributions
    # which means minimizing negative log likelihood is equivalent to minimizing l1 distance.
    # TODO: Is it still ok with the tanh activation?
    return tf.reduce_mean(tf.abs(x_rec - x))


def vae_loss(x, encoder, decoder, **hparams):
    z_mu = encoder(x)
    x_reconstructed = decoder(z_mu + tf.random_normal(tf.shape(z_mu), stddev=1.0))
    return hparams["lambda_vae_kl"] * kl_divergence(z_mu) + \
           hparams["lambda_vae_nll"] * nll(x, x_reconstructed)


def gan_losses(d_real_linear, d_fake_linear, **hparams):

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_linear,
        labels=tf.ones_like(d_fake_linear)))

    discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.concat((d_real_linear, d_fake_linear), axis=1),
        labels=tf.concat((tf.ones_like(d_real_linear), tf.zeros_like(d_fake_linear)), axis=1)))

    return hparams["lambda_gan"] * discriminator_loss, hparams["lambda_gan"] * generator_loss


def lsgan_losses():
    pass  # TODO


def cyclic_loss(x, encoder, decoder, other_encoder, other_decoder, **hparams):
    z_mu = encoder(x)

    translation = other_decoder(z_mu + tf.random_normal(tf.shape(z_mu), stddev=1.0))
    z_mu_translation = other_encoder(translation)

    x_cycle_reconstruction = decoder(z_mu_translation + tf.random_normal(tf.shape(z_mu), stddev=1.0))

    return hparams["lambda_cyclic_kl"] * kl_divergence(z_mu) + \
           hparams["lambda_cyclic_kl"] * kl_divergence(z_mu_translation) + \
           hparams["lambda_cyclic_nll"] * nll(x, x_cycle_reconstruction)
