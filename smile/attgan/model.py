import tensorflow as tf

from smile.attgan.loss import classification_loss, wgan_gp_losses
from smile.utils.tf_utils import img_summary

"""
def preprocess(x):
    h, w = x.shape[1:-1]
    x = x * 2 - 1
    x = tf.image.resize_images(x, [h - 2, w - 2])
    return x


def postprocess(x):
    h, w = x.shape[1:-1]
    x = tf.image.resize_images(x, [h + 2, w + 2])
    x = (x + 1) / 2
    return x
"""


def preprocess(x):
    x = tf.image.crop_to_bounding_box(x, 26, 3, 170, 170)
    x = tf.image.resize_images(x, (128, 128))
    x = x * 2 - 1
    return x


class AttGAN:
    def __init__(self,
                 img,
                 attributes,
                 img_test,
                 attributes_test,
                 encoder_fn,
                 decoder_fn,
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 **hparams):
        is_training = tf.placeholder_with_default(False, [])

        _, n_classes = attributes.get_shape()

        # TODO: preprocess/postprocess [-1, 1] etc

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
            # Existing ones shouldnt happen.
            # sample num ones per row as well.
            # sampled_attributes = \
            #    tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)
            return tf.cast(tf.logical_not(tf.cast(attributes, tf.bool)), tf.float32)  # Just invert the attributes.

        x = preprocess(img)
        z = encoder(x)  # TODO: Include attribute intensity part.
        sampled_attributes = generate_attributes(attributes)
        x_translated = decoder(z, sampled_attributes)
        x_reconstructed = decoder(z, attributes)

        encoder_decoder_classification_loss = classification_loss(sampled_attributes, classifier(x_translated))
        classifier_classification_loss = classification_loss(attributes, classifier(x))

        reconstruction_loss = tf.reduce_mean(tf.abs(x - x_reconstructed))

        discriminator_adversarial_loss, encoder_decoder_adversarial_loss = \
            wgan_gp_losses(x, x_translated, discriminator)

        encoder_decoder_loss = (hparams["lambda_rec"] * reconstruction_loss +
                                hparams["lambda_cls_g"] * encoder_decoder_classification_loss +
                                encoder_decoder_adversarial_loss)

        discriminator_classifier_loss = (hparams["lambda_cls_d"] * classifier_classification_loss +
                                         discriminator_adversarial_loss)

        learning_rate = 1e-4  # TODO: with decay?

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        global_step = tf.train.get_or_create_global_step()

        disc_cls_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="(classifier_discriminator_shared|discriminator_private|classifier_private)")
        enc_dec_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="(encoder|decoder)")

        optimization_step = tf.group(
            create_update_step(encoder_decoder_loss, enc_dec_variables),
            create_update_step(discriminator_classifier_loss, disc_cls_variables),
            global_step.assign_add(1))

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
                tf.reduce_mean(tf.metrics.accuracy(attributes, tf.sigmoid(classifier(x)) > 0))),
            tf.summary.scalar(
                "cls/mean_accuracy_fake",
                tf.reduce_mean(tf.metrics.accuracy(sampled_attributes, tf.sigmoid(classifier(x_translated)) > 0))),

            tf.summary.scalar("learning_rate", learning_rate)
        ))

        # TODO: For test image, visualize translations for all attributes we train for. Both single and multiple.
        # TODO: Add visualizations for sliding intensity.

        x_test = preprocess(img_test)
        image_summaries = tf.summary.merge((
            img_summary("train", x, x_translated),
            img_summary("test", x_test, decoder(encoder(x_test), generate_attributes(attributes_test)))
        ))

        self.is_training = is_training
        self.global_step = global_step
        self.train_op = optimization_step
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

    def train_step(self, sess, summary_writer):
        _, scalar_summaries, i = sess.run(
            (self.train_op, self.scalar_summaries, self.global_step),
            feed_dict={self.is_training: True})
        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def export(self, sess, export_dir):
        pass
