import tensorflow as tf

from smile.models.attgan.loss import classification_loss, lsgan_losses, wgan_gp_losses
from smile.utils.tf_utils import img_summary_with_text


def preprocess(x):
    # TODO: Add crop and resize to data pipeline instead.
    x = tf.image.crop_to_bounding_box(x, 26, 3, 170, 170)
    x = tf.image.resize_images(x, (128, 128))
    x = x * 2 - 1
    return x


def postprocess(x):
    return (x + 1) / 2


class AttGAN:
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test,
                 encoder_fn,
                 decoder_fn,
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 **hparams):

        is_training = tf.placeholder_with_default(False, [])

        _, n_classes = attributes.get_shape()

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
            # TODO: Maybe need better way of sampling target attributes
            # Existing ones shouldn't be part of target attributes?
            # Maybe sample number of active attributes per instance as well?
            # Or just randomly shuffle attributes

            # 50/50 sample per attribute.
            sampled_attributes = \
                tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)

            # Just invert every attribute.
            #sampled_attributes = tf.cast(tf.logical_not(tf.cast(attributes, tf.bool)), tf.float32)

            return sampled_attributes

        x = preprocess(img)
        z = encoder(x)  # TODO: Include attribute intensity part.
        sampled_attributes = generate_attributes(attributes)
        x_translated = decoder(z, sampled_attributes)
        x_reconstructed = decoder(z, attributes)

        encoder_decoder_classification_loss = classification_loss(sampled_attributes, classifier(x_translated))
        classifier_classification_loss = classification_loss(attributes, classifier(x))

        reconstruction_loss = tf.reduce_mean(tf.abs(x - x_reconstructed))

        if hparams["adversarial_loss_type"] == "wgan-gp":
            discriminator_adversarial_loss, encoder_decoder_adversarial_loss = \
                wgan_gp_losses(x, x_translated, discriminator)
        elif hparams["adversarial_loss_type"] == "lsgan":
            discriminator_adversarial_loss, encoder_decoder_adversarial_loss = \
                lsgan_losses(x, x_translated, discriminator)
        else:
            raise ValueError("Invalid adversarial loss type.")

        encoder_decoder_loss = (hparams["lambda_rec"] * reconstruction_loss +
                                hparams["lambda_cls_g"] * encoder_decoder_classification_loss +
                                encoder_decoder_adversarial_loss)

        discriminator_classifier_loss = (hparams["lambda_cls_d"] * classifier_classification_loss +
                                         discriminator_adversarial_loss)

        # TODO: Classifier needs regularization.
        # Or early stopping (stop updating this after a while)

        global_step = tf.train.get_or_create_global_step()

        initial_learning_rate = 2e-4
        dataset_size = 180000
        steps_per_epoch = dataset_size // hparams["batch_size"]
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=100 * steps_per_epoch,
            decay_rate=0.1,
            staircase=True)

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        disc_cls_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="(classifier|discriminator)")
        enc_dec_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="(encoder|decoder)")
        assert (set(disc_cls_variables) & set(enc_dec_variables)) == set(), "D and G should not share variables."

        enc_dec_train_step = create_update_step(encoder_decoder_loss, enc_dec_variables)
        disc_cls_train_step = create_update_step(discriminator_classifier_loss, disc_cls_variables)

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/enc_dec_loss", encoder_decoder_loss),
            tf.summary.scalar("loss/disc_cls_loss", discriminator_classifier_loss),
            tf.summary.scalar("loss/parts/cls_classification_loss", classifier_classification_loss),
            tf.summary.scalar("loss/parts/enc_dec_classification_loss", encoder_decoder_classification_loss),
            tf.summary.scalar("loss/parts/enc_dec_adversarial_loss", encoder_decoder_adversarial_loss),
            tf.summary.scalar("loss/parts/disc_adversarial_loss", discriminator_adversarial_loss),
            tf.summary.scalar("loss/parts/enc_dec_reconstruction_loss", reconstruction_loss),

            tf.summary.scalar("disc/real", tf.reduce_mean(discriminator(x))),
            tf.summary.scalar("disc/fake", tf.reduce_mean(discriminator(x_translated))),

            tf.summary.scalar(
                "cls/mean_accuracy_real",
                tf.reduce_mean(tf.metrics.accuracy(attributes, tf.sigmoid(classifier(x)) > 0.5))),
            tf.summary.scalar(
                "cls/mean_accuracy_fake",
                tf.reduce_mean(tf.metrics.accuracy(sampled_attributes, tf.sigmoid(classifier(x_translated)) > 0.5))),

            tf.summary.scalar("learning_rate", learning_rate)
        ))

        # TODO: Add visualizations for sliding intensity.

        x_test = preprocess(img_test)
        sampled_attributes_test = generate_attributes(attributes_test)
        x_test_translated = decoder(encoder(x_test, is_training=False), sampled_attributes_test, is_training=False)
        image_summaries = tf.summary.merge((
            img_summary_with_text("train", attribute_names,
                                  postprocess(x), attributes,
                                  postprocess(x_translated), sampled_attributes),

            img_summary_with_text("test", attribute_names,
                                  postprocess(x_test), attributes_test,
                                  postprocess(x_test_translated), sampled_attributes_test)
        ))

        self.is_training = is_training
        self.global_step = global_step
        self.global_step_increment = global_step.assign_add(1)
        self.enc_dec_train_step = enc_dec_train_step
        self.disc_cls_train_step = disc_cls_train_step
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

    def train_step(self, sess, summary_writer, **hparams):
        n_discriminator_iters = 5 if hparams["adversarial_loss_type"] == "wgan-gp" else 1
        for _ in range(n_discriminator_iters):
            sess.run(self.disc_cls_train_step, feed_dict={self.is_training: True})

        _, scalar_summaries, i = sess.run(
            (self.enc_dec_train_step, self.scalar_summaries, self.global_step_increment),
            feed_dict={self.is_training: True})

        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def export(self, sess, export_dir):
        pass
