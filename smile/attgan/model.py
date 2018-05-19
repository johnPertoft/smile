import tensorflow as tf

from smile.attgan.loss import wgan_gp_losses


class AttGAN:
    def __init__(self,
                 img,
                 attributes,
                 encoder_fn,
                 decoder_fn,
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 **hparams):

        is_training = tf.placeholder_with_default(False, [])
        x_test = None  # TODO: input pipeline with test images.

        _cd_shared = tf.make_template("classifier_discriminator_shared", classifier_discriminator_shared_fn, **hparams)
        _d_private = tf.make_template("discriminator_private", discriminator_private_fn)
        _c_private = tf.make_template("classifier_private", classifier_private_fn)
        cd_scope_regex = "(classifier_discriminator_shared|discriminator_private|classifier_private)"

        # Model parts.
        encoder = tf.make_template("encoder", encoder_fn, **hparams)
        decoder = tf.make_template("decoder", decoder_fn, **hparams)
        classifier = lambda x: _c_private(_cd_shared(x))
        discriminator = lambda x: _d_private(_cd_shared(x))

        x = img
        z = encoder(x)  # TODO: Include attribute intensity part.
        sampled_attributes = None  # TODO: How to sample this for training? TODO: Placeholder
        x_translated = decoder(z, sampled_attributes)
        x_reconstructed = decoder(z, attributes)

        # Losses.

        classification_loss = lambda t, l: tf.reduce_mean(tf.losses.sigmoid_cross_entropy(t, l))
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

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        learning_rate = 1e-4  # TODO: with decay?

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        global_step = tf.train.get_or_create_global_step()

        optimization_step = tf.group(
            create_update_step(encoder_decoder_loss, get_vars("(encoder|decoder)")),
            create_update_step(discriminator_classifier_loss, get_vars(cd_scope_regex)),
            global_step.assign_add(1)
        )

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/enc_dec_loss", encoder_decoder_loss),
            tf.summary.scalar("loss/disc_cls_loss", discriminator_classifier_loss),
            tf.summary.scalar("loss/parts/cls_classification_loss", classifier_classification_loss),
            tf.summary.scalar("loss/parts/enc_dec_classification_loss", encoder_decoder_classification_loss),
            tf.summary.scalar("loss/parts/enc_dec_adversarial_loss", encoder_decoder_adversarial_loss),
            tf.summary.scalar("loss/parts/disc_adversarial_loss", discriminator_adversarial_loss),
            tf.summary.scalar("loss/parts/enc_dec_reconstruction_loss", reconstruction_loss),
            tf.summary.scalar("disc/real", discriminator(x)),
            tf.summary.scalar("disc/fake", discriminator(x_translated)),
            tf.summary.scalar(
                "cls/mean_accuracy_real",
                tf.reduce_mean(tf.metrics.accuracy(attributes, tf.sigmoid(classifier(x)) > 0))),
            tf.summary.scalar(
                "cls/mean_accuracy_fake",
                tf.reduce_mean(tf.metrics.accuracy(sampled_attributes, tf.sigmoid(classifier(x_translated)) > 0))),
            tf.summary.scalar("learning_rate", learning_rate)
        ))

        def img_summary(name, before, after):
            # TODO: Need to visualize, sampled_attributes as well.
            side_by_side = tf.concat((before, after), axis=2)[:3]
            return tf.summary.image(name, side_by_side)

        # TODO: For test image, visualize translations for all attributes we train for.
            # Both single and multiple.
        # TODO: Add visualizations for sliding intensity.
        image_summaries = tf.summary.merge((
            img_summary("train", x, x_translated),
            img_summary("test", x_test, decoder(encoder(x_test), sampled_attributes))
        ))

        self.is_training = is_training
        self.global_step = global_step
        self.train_step = optimization_step
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

    def train_step(self):
        pass

    def export(self):
        pass
