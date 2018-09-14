import numpy as np
import skimage.io
import tensorflow as tf

from smile.experiments.samples import multi_attribute_translation_samples
from smile.experiments.summaries import img_summary_with_text
from smile.models import Model


class AttGAN(Model):
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test,
                 img_test_static, attributes_test_static,
                 encoder_fn,
                 decoder_fn,
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 adversarial_loss_fn,
                 **hparams):

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        # TODO: Remove this default value. Specify on each invocation instead?
        is_training = tf.placeholder_with_default(False, [])

        n_classes = attributes.shape[1].value

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
            n_classes=n_classes,
            is_training=is_training,
            **hparams)

        # Model parts.
        encoder = tf.make_template("encoder", encoder_fn, is_training=is_training, **hparams)
        decoder = tf.make_template("decoder", decoder_fn, is_training=is_training, **hparams)
        classifier = lambda x: _c_private(_cd_shared(x))
        discriminator = lambda x: _d_private(_cd_shared(x))

        def generate_attributes(attributes):
            # 50/50 sample per attribute.
            return tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)
            # Just invert every attribute.
            #return tf.cast(tf.logical_not(tf.cast(attributes, tf.bool)), tf.float32)

        x = preprocess(img)
        z = encoder(x)  # TODO: Include attribute intensity part.
        target_attributes = generate_attributes(attributes)
        x_translated = decoder(z, target_attributes)
        x_reconstructed = decoder(z, attributes)
        target_attributes_test = generate_attributes(attributes_test)
        x_test_translated = decoder(encoder(preprocess(img_test), is_training=False),
                                    target_attributes_test, is_training=False)

        # Loss parts.
        disc_adversarial_loss, enc_dec_adversarial_loss = adversarial_loss_fn(x, x_translated, discriminator, **hparams)
        cls_real_loss = tf.losses.sigmoid_cross_entropy(attributes, classifier(x))
        cls_fake_loss = tf.losses.sigmoid_cross_entropy(target_attributes, classifier(x_translated))
        reconstruction_loss = tf.losses.absolute_difference(x, x_reconstructed)

        # Full objectives.
        enc_dec_loss = hparams["lambda_rec"] * reconstruction_loss + \
                       hparams["lambda_cls_g"] * cls_fake_loss + \
                       enc_dec_adversarial_loss
        disc_cls_loss = hparams["lambda_cls_d"] * cls_real_loss + disc_adversarial_loss

        # TODO: Add regularization to classifier.
        # Or try early stopping (stop updating this after a while).
        # TODO: Fix learning rate decay.

        global_step = tf.train.get_or_create_global_step()

        # TODO: Make sure this is correct.
        initial_learning_rate = 2e-4
        dataset_size = 180000
        steps_per_epoch = dataset_size // hparams["batch_size"]
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=100 * steps_per_epoch,
            decay_rate=0.1,
            staircase=True)

        disc_cls_variables = tf.trainable_variables("(classifier|discriminator)")
        enc_dec_variables = tf.trainable_variables("(encoder|decoder)")
        assert (set(disc_cls_variables) & set(enc_dec_variables)) == set(), "D and G should not share variables."

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        enc_dec_train_step = create_update_step(enc_dec_loss, enc_dec_variables)
        disc_cls_train_step = create_update_step(disc_cls_loss, disc_cls_variables)

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/enc_dec_loss", enc_dec_loss),
            tf.summary.scalar("loss/disc_cls_loss", disc_cls_loss),
            tf.summary.scalar("loss/parts/real_classification_loss", cls_real_loss),
            tf.summary.scalar("loss/parts/fake_classification_loss", cls_fake_loss),
            tf.summary.scalar("loss/parts/enc_dec_adversarial_loss", enc_dec_adversarial_loss),
            tf.summary.scalar("loss/parts/disc_adversarial_loss", disc_adversarial_loss),
            tf.summary.scalar("loss/parts/enc_dec_reconstruction_loss", reconstruction_loss),

            tf.summary.scalar("disc/real", tf.reduce_mean(discriminator(x))),
            tf.summary.scalar("disc/fake", tf.reduce_mean(discriminator(x_translated))),

            tf.summary.scalar(
                "cls/mean_accuracy_real",
                tf.reduce_mean(tf.metrics.accuracy(attributes, tf.sigmoid(classifier(x)) > 0.5))),
            tf.summary.scalar(
                "cls/mean_accuracy_fake",
                tf.reduce_mean(tf.metrics.accuracy(target_attributes, tf.sigmoid(classifier(x_translated)) > 0.5))),

            tf.summary.scalar("learning_rate", learning_rate)
        ))

        # TODO: Add visualizations for sliding intensity. (Both to summaries and samples.)

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
        self.enc_dec_train_step = enc_dec_train_step
        self.disc_cls_train_step = disc_cls_train_step
        self.n_discriminator_iters = hparams["n_discriminator_iters"]
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

        # Handles for progress samples.
        self.translation_samples = multi_attribute_translation_samples(
            img_test_static,
            attributes_test_static,
            lambda x, a: postprocess(decoder(encoder(preprocess(x), is_training=False), a, is_training=False)))

        # Handles for exporting.
        # TODO:

    def train_step(self, sess, summary_writer):
        for _ in range(self.n_discriminator_iters):
            sess.run(self.disc_cls_train_step, feed_dict={self.is_training: True})

        _, i = sess.run((self.enc_dec_train_step, self.global_step_increment), feed_dict={self.is_training: True})

        scalar_summaries = sess.run(self.scalar_summaries)
        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def generate_samples(self, sess, fname):
        img = sess.run(self.translation_samples)
        img = np.vstack([np.hstack(x) for x in img])
        skimage.io.imsave(fname, img)
        # TODO: Add attribute names for rows.

    def export(self, sess, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        # TODO: Add common export function.

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "translate"
            })

        return builder.save()
