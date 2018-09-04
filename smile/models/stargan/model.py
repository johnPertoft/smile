import tensorflow as tf

from smile.models import Model
from smile.models.stargan.loss import lsgan_losses, attribute_classification_losses


def concat_attributes(x, attributes):
    """Depthwise concatenation of image and attributes vector."""
    c = attributes[:, tf.newaxis, tf.newaxis, :]
    h, w = x.get_shape()[1:3]
    c = tf.tile(c, (1, h, w, 1))
    return tf.concat((x, c), axis=3)


class StarGAN(Model):
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test,
                 img_test_static, attributes_test_static,
                 generator_fn,
                 discriminator_fn,
                 **hparams):

        # TODO: Fix this implementation.

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        is_training = tf.placeholder_with_default(False, [])

        imgs = preprocess(imgs)
        _, n_attributes = attributes.get_shape()

        generator = tf.make_template("generator", generator_fn, is_training=is_training)
        discriminator = tf.make_template("discriminator", discriminator_fn,
                                         n_attributes=n_attributes, is_training=is_training)

        # TODO: Generate target_attributes in a better way. I.e. original attribute shouldn't be there.
        # TODO: Some attributes should also be mutually exclusive, like hair colors.
        target_attributes = \
            tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)
        translated_imgs = generator(concat_attributes(imgs, target_attributes))

        # Adversarial loss.
        d_real, d_real_predicted_attributes = discriminator(imgs)
        d_fake, d_fake_predicted_attributes = discriminator(translated_imgs)
        d_adversarial_loss, g_adversarial_loss = lsgan_losses(d_real, d_fake)  # TODO: Paper uses wgan loss.
        d_classification_loss, g_classification_loss = attribute_classification_losses(
            d_real_predicted_attributes, attributes,
            d_fake_predicted_attributes, target_attributes)

        # Reconstruction loss.
        reconstructed_imgs = generator(concat_attributes(translated_imgs, attributes))
        reconstruction_loss = tf.reduce_mean(tf.abs(imgs - reconstructed_imgs))

        # Full objective.
        d_loss = d_adversarial_loss + hparams["lambda_cls"] * d_classification_loss
        g_loss = g_adversarial_loss + hparams["lambda_cls"] * g_classification_loss + hparams["lambda_rec"] * reconstruction_loss

        global_step = tf.train.get_or_create_global_step()

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        d_update_step = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=get_vars("discriminator"))
        g_update_step = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=get_vars("generator"))

        train_step = tf.group(d_update_step, g_update_step, global_step.assign_add(1))

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("d_loss", d_loss),
            tf.summary.scalar("g_loss", g_loss)
        ))

        # TODO: Add target attributes to image summaries.
        image_summaries = tf.summary.image("A_to_B", postprocess(tf.concat((imgs[:3], translated_imgs[:3]), axis=2)))

        self.train_op = train_step
        self.global_step = global_step
        self.is_training = is_training
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

    def train_step(self, sess, summary_writer):
        feed_dict = {
            self.is_training: True
        }

        _, scalar_summaries, i = sess.run((self.train_op, self.scalar_summaries, self.global_step), feed_dict=feed_dict)

        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

    def export(self, sess, export_dir):
        pass
