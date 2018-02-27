import tensorflow as tf

# TODO
# normal gan loss
# "stable" gan loss


def lsgan_losses(D_real, D_fake):

    D_real_loss = tf.reduce_mean((D_real - 1.0) ** 2.0)
    D_fake_loss = tf.reduce_mean(D_fake ** 2.0)
    D_loss = (D_real_loss + D_fake_loss) / 2.0

    G_loss = tf.reduce_mean((D_fake - 1.0) ** 2.0)

    return D_loss, G_loss


def wgan_gp_losses(C_real, C_fake, X_sampled, X_fake, critic_fn, wgan_lambda):
    if C_real.shape.ndims > 1:
        axis = list(range(1, C_real.shape.ndims))
        C_real = tf.reduce_mean(C_real, axis=axis)
        C_fake = tf.reduce_mean(C_fake, axis=axis)

    # Generator loss.
    G_loss = -tf.reduce_mean(C_fake)  # Maximize critic output for fake samples.

    # Gradient penalty.
    epsilon = tf.random_uniform(shape=tf.shape(X_fake)[:1], minval=0.0, maxval=1.0)
    for _ in range(X_fake.shape.ndims - 1):
        epsilon = epsilon[:, tf.newaxis]
    X_interpolate = epsilon * X_sampled + (1.0 - epsilon) * X_fake
    critic_X_interpolate = tf.reduce_mean(critic_fn(X_interpolate), axis=[1, 2, 3])
    C_X_interpolate_grads = tf.gradients(critic_X_interpolate, X_interpolate)[0]
    C_X_interpolate_grads_norm = tf.norm(C_X_interpolate_grads, ord=2, axis=1)
    gradient_penalty = wgan_lambda * tf.reduce_mean(tf.square(C_X_interpolate_grads_norm - 1.0))

    # Discriminator loss.
    D_real_loss = -tf.reduce_mean(C_real)  # Maximize critic output for real samples.
    D_fake_loss = tf.reduce_mean(C_fake)  # Minimize critic output for fake samples.
    D_loss = D_real_loss + D_fake_loss + gradient_penalty

    return D_loss, G_loss
