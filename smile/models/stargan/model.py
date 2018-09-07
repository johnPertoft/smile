import tensorflow as tf

from smile.experiments.samples import multi_attribute_translation_samples
from smile.experiments.summaries import img_summary_with_text
from smile.models import Model


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
                 adversarial_loss_fn,
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

        generator = tf.make_template("generator", generator_fn, is_training=is_training)
        classifier = lambda x: _c_private(_cd_shared(x))
        discriminator = lambda x: _d_private(_cd_shared(x))

        def generate_attributes(attributes):
            # 50/50 sample per attribute.
            return tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)

        x = preprocess(img)
        target_attributes = generate_attributes(attributes)
        x_translated = generator(x, target_attributes)
        x_reconstructed = generator(x_translated, attributes)
        target_attributes_test = generate_attributes(attributes_test)
        x_test_translated = generator(preprocess(img_test), target_attributes_test)

        # Loss parts.
        d_adversarial_loss, g_adversarial_loss = adversarial_loss_fn(x, x_translated, discriminator, **hparams)
        real_classification_loss = tf.losses.sigmoid_cross_entropy(attributes, classifier(x))
        fake_classification_loss = tf.losses.sigmoid_cross_entropy(target_attributes, classifier(x_translated))
        reconstruction_loss = tf.losses.absolute_difference(x, x_reconstructed)

        # Full objectives.
        d_loss = d_adversarial_loss + hparams["lambda_cls"] * real_classification_loss
        g_loss = g_adversarial_loss + hparams["lambda_cls"] * fake_classification_loss + \
                 hparams["lambda_rec"] * reconstruction_loss

        d_vars = tf.trainable_variables("(classifier|discriminator)")
        g_vars = tf.trainable_variables("generator")
        assert (set(d_vars) & set(g_vars)) == set(), "D and G should not share variables."

        # TODO: Should decay after 10 epochs.
        learning_rate = 1e-4

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        d_update_step = create_update_step(d_loss, d_vars)
        g_update_step = create_update_step(g_loss, g_vars)

        global_step = tf.train.get_or_create_global_step()

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/d", d_loss),
            tf.summary.scalar("loss/g", g_loss),
            tf.summary.scalar("loss/d_adv", d_adversarial_loss),
            tf.summary.scalar("loss/g_adv", g_adversarial_loss),
            tf.summary.scalar("loss/real_cls", real_classification_loss),
            tf.summary.scalar("loss/fake_cls", fake_classification_loss),
            tf.summary.scalar("loss/rec", reconstruction_loss),

            # TODO: Add d_real and d_fake
            # TODO: Add accuracy for classifier

        ))

        image_summaries = tf.summary.merge((
            img_summary_with_text(
                "train",
                attribute_names,
                img, attributes,
                postprocess(x_translated), target_attributes),

            img_summary_with_text(
                "test",
                attribute_names,
                img_test, attributes_test,
                postprocess(x_test_translated), target_attributes_test)
        ))

        # Handles for training.
        self.is_training = is_training
        self.global_step_increment = global_step.assign_add(1)
        self.d_update_step = d_update_step
        self.g_update_step = g_update_step
        self.n_discriminator_iters = hparams["n_discriminator_iters"]
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

        # Handles for progress samples.
        self.translation_samples = multi_attribute_translation_samples(
            img_test_static,
            attributes_test_static,
            lambda x, a: postprocess(generator(preprocess(x), a)))

        # Handles for exporting.
        # TODO:

    def train_step(self, sess, summary_writer):
        for _ in range(self.n_discriminator_iters):
            sess.run(self.d_update_step, feed_dict={self.is_training: True})

        _, scalar_summaries, i = sess.run(
            (self.g_update_step, self.scalar_summaries, self.global_step_increment),
            feed_dict={self.is_training: True})

        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def export(self, sess, export_dir):
        pass
