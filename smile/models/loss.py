import tensorflow as tf

# TODO
# normal gan loss
# "stable" gan loss
# wasserstein gan loss
# TODO: Also see tf built in gan losses.


# TODO: Should make sure that we reduce the patch critic correctly? I.e. first by each sample and then by batch

def lsgan_losses(D_real, D_fake):

    D_real_loss = tf.reduce_mean((D_real - 1.0) ** 2.0)
    D_fake_loss = tf.reduce_mean(D_fake ** 2.0)
    D_loss = (D_real_loss + D_fake_loss) / 2.0

    G_loss = tf.reduce_mean((D_fake - 1.0) ** 2.0)

    return D_loss, G_loss


def wgan_gp_losses(D_real, D_fake, X_sampled, X_fake, critic_fn, wgan_lambda):
    #if not scalar reduce within samples
    # hmm, not sure
    D_real = tf.reduce_mean(D_real, axis=[1, 2, 3])
    D_fake = tf.reduce_mean(D_fake, axis=[1, 2, 3])

    # Generator loss.
    G_loss = -tf.reduce_mean(D_fake)  # Maximize critic output for fake samples.

    # Gradient penalty.
    # TODO: refactor this a bit
    epsilon_shape = tf.concat((tf.shape(X_fake)[:1], tf.ones_like(tf.shape(X_fake)[1:])), axis=0)
    epsilon = tf.random_uniform(shape=epsilon_shape, minval=0.0, maxval=1.0)
    X_interpolated = epsilon * X_sampled + (1.0 - epsilon) * X_fake
    C_X_interpolated_grads = tf.gradients(critic_fn(X_interpolated), X_interpolated)[0]  # TODO: need reduce_mean
    C_X_interpolated_grads_norm = tf.norm(C_X_interpolated_grads, ord=2, axis=1)
    gradient_penalty = wgan_lambda * tf.reduce_mean(tf.square(C_X_interpolated_grads_norm - 1.0))

    # Discriminator loss.
    D_real_loss = -tf.reduce_mean(D_real)  # Maximize critic output for real samples.
    D_fake_loss = tf.reduce_mean(D_fake)  # Minimize critic output for fake samples.
    D_loss = D_real_loss + D_fake_loss + gradient_penalty

