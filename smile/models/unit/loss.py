import tensorflow as tf


def kl_divergence(z_mu):
    # Hard coded kl divergence with unit normal.
    prior = tf.distributions.Normal(loc=tf.zeros_like(z_mu), scale=tf.ones_like(z_mu))
    q_z_x = tf.distributions.Normal(loc=z_mu, scale=tf.ones_like(z_mu))
    kl_div = tf.distributions.kl_divergence(q_z_x, prior)
    return tf.reduce_mean(kl_div)


def vae_loss(x, encoder, decoder, sample_z, **hparams):
    z_mu = encoder(x)
    x_reconstructed = decoder(sample_z(z_mu))
    return hparams["lambda_vae_kl"] * kl_divergence(z_mu) + \
           hparams["lambda_vae_rec"] * tf.losses.absolute_difference(x, x_reconstructed)


def cyclic_loss(x, encoder_1, decoder_1, encoder_2, decoder_2, sample_z, **hparams):
    z_mu = encoder_1(x)
    x_translated = decoder_2(sample_z(z_mu))
    z_mu_translated = encoder_2(x_translated)
    x_reconstructed = decoder_1(sample_z(z_mu_translated))
    return hparams["lambda_cyclic_kl"] * kl_divergence(z_mu) + \
           hparams["lambda_cyclic_kl"] * kl_divergence(z_mu_translated) + \
           hparams["lambda_cyclic_rec"] * tf.losses.absolute_difference(x, x_reconstructed)
