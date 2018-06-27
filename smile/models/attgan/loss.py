import tensorflow as tf


def classification_loss(targets, logits):
    return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(targets, logits))


def wgan_gp_losses(x_real, x_fake, critic):
    # Interpolate between x_real and x_fake.
    shape = [tf.shape(x_real)[0]] + [1] * (x_real.shape.ndims - 1)
    epsilon = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
    x_interpolate = epsilon * x_real + (1.0 - epsilon) * x_fake

    # Gradient penalty.
    predictions = critic(x_interpolate)
    gradients = tf.gradients(predictions, x_interpolate)[0]
    norm = tf.norm(tf.layers.flatten(gradients), axis=1)
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2.0)  # TODO: Try with l1 norm as well?

    # Losses for critic and generator.
    c_real = critic(x_real)
    c_fake = critic(x_fake)
    critic_loss = -tf.reduce_mean(c_real) + tf.reduce_mean(c_fake) + gradient_penalty * 10.0
    generator_loss = -tf.reduce_mean(c_fake)

    return critic_loss, generator_loss


def lsgan_losses(x_real, x_fake, discriminator):
    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)

    d_real_loss = tf.reduce_mean((d_real - 1.0) ** 2.0)
    d_fake_loss = tf.reduce_mean(d_fake ** 2.0)
    d_loss = (d_real_loss + d_fake_loss) / 2.0

    g_loss = tf.reduce_mean((d_fake - 1.0) ** 2.0)

    return d_loss, g_loss