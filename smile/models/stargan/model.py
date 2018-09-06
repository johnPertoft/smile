import tensorflow as tf

from smile.losses import lsgan_losses
from smile.models import Model


# TODO: Put this in architecture file.
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
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 **hparams):

        # TODO: Fix this implementation.
        # TODO: Add support for training with facial expression dataset at the same time.
            # Mask vector implementation etc.

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        is_training = tf.placeholder_with_default(False, [])

        n_attributes = attributes.shape[1].value

        _cd_shared = tf.make_template(
            "classifier_discriminator_shared",
            classifier_discriminator_shared_fn,
            is_training=is_training,
            **hparams)
        _d_private = tf.make_template(
            "discriminator_private",
            discriminator_private_fn,
            is_training=is_training,
            **hparams)
        _c_private = tf.make_template(
            "classifier_private",
            classifier_private_fn,
            n_attributes=n_attributes,
            is_training=is_training,
            **hparams)

        # Model parts.
        generator = tf.make_template("generator", generator_fn, is_training=is_training)
        classifier = lambda x: _c_private(_cd_shared(x))
        discriminator = lambda x: _d_private(_cd_shared(x))

        def generate_attributes(attributes):
            # TODO
            #target_attributes = \
            #   tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)
            pass

        x = preprocess(img)
        target_attributes = generate_attributes(attributes)
        x_translated = generator(x, target_attributes)
        x_reconstructed = generator(x_translated, attributes)

        # TODO: Paper uses wgan-gp loss.
        # TODO: Take adversarial loss fn as input.
        d_adversarial_loss, g_adversarial_loss = lsgan_losses(x, x_translated, discriminator)

        d_classification_loss = tf.losses.sigmoid_cross_entropy(attributes, classifier(x))
        g_classification_loss = tf.losses.sigmoid_cross_entropy(target_attributes, classifier(x_translated))

        reconstruction_loss = tf.losses.absolute_difference(x, x_reconstructed)

        # Full objectives.
        d_loss = d_adversarial_loss + hparams["lambda_cls"] * d_classification_loss
        g_loss = g_adversarial_loss + hparams["lambda_cls"] * g_classification_loss \
                 + hparams["lambda_rec"] * reconstruction_loss

        global_step = tf.train.get_or_create_global_step()

        tvars = lambda scope: tf.trainable_variables(scope)
        d_update_step = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=tvars("(classifier|discriminator)"))
        g_update_step = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=tvars("generator"))

        # TODO: Don't group if using wgan loss.
        # TODO: Potentially run separately anyway?
        train_step = tf.group(d_update_step, g_update_step, global_step.assign_add(1))

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("d_loss", d_loss),
            tf.summary.scalar("g_loss", g_loss)
        ))

        # TODO: Fix summaries.

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
