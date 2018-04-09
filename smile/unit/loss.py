import tensorflow as tf


def vae_loss(x, x_rec, **hparams):
    # Assuming a gaussian encoding distribution with unit variance.
    # kl(posterior||prior) = log(sigma_2/sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2*sigma_2**2) - 0.5
    # TODO: maybe use tf.distributions.kl_divergence
    kl_loss = 0.5 * (1.0 + tf.square(mu_1 - mu_2)) - 0.5
