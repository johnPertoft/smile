import tensorflow as tf


def patches_to_scalar(x):
    # TODO: Check if necessary first.
    # TODO: rename
    return tf.reduce_mean(x, axis=[1, 2])


def attribute_classification_losses(d_real_predicted_attributes,
                                    actual_attributes,
                                    d_fake_predicted_attributes,
                                    target_attributes):

    # TODO: Do we want to do this even?
    d_real_predicted_attributes = patches_to_scalar(d_real_predicted_attributes)
    d_fake_predicted_attributes = patches_to_scalar(d_fake_predicted_attributes)

    d_classification_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=actual_attributes,
                                                                                   logits=d_real_predicted_attributes))

    g_classification_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_attributes,
                                                                                   logits=d_fake_predicted_attributes))

    return d_classification_loss, g_classification_loss


def lsgan_losses(D_real, D_fake):
    """Least Squares GAN loss."""

    # TODO: reduce patchgan outputs first.

    D_real_loss = tf.reduce_mean((D_real - 1.0) ** 2.0)
    D_fake_loss = tf.reduce_mean(D_fake ** 2.0)
    D_loss = (D_real_loss + D_fake_loss) / 2.0

    G_loss = tf.reduce_mean((D_fake - 1.0) ** 2.0)

    return D_loss, G_loss
