import tensorflow as tf


def vae_loss(x, x_rec, z, **hparams):
    # KL divergence.
    prior = tf.distributions.Normal(loc=tf.zeros_like(z), scale=tf.ones_like(z))
    q_z_x = tf.distributions.Normal(loc=z, scale=tf.ones_like(z))
    kl_div = tf.distributions.kl_divergence(q_z_x, prior)

    # TODO: Should minimize nll of p(x|z)
    # TODO: We probably want decoders to have a tanh activation? How to write negative log likelihood for this?
    # TODO: Paper says that the decoder models a laplacian, should we have linear output?
    # TODO: Minimizing nll for isotropic laplacian is equivalent to minimizing l1 norm of difference.
    # Reconstruction losses for VAEs assuming Laplace distribution.
    reconstruction_loss = tf.reduce_mean(tf.abs(x_rec - x))

    vae_loss = hparams["lambda_kl"] * tf.reduce_mean(kl_div) + hparams["lambda_nll"] * reconstruction_loss

    return vae_loss


def gan_losses(d_real, d_fake, **hparams):
    pass
