import tensorflow as tf

from smile.models import Model


class SGGAN(Model):
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test,
                 img_test_static, attributes_test_static,
                 encoder_fn,
                 bottleneck_fn,
                 decoder_fn,
                 classifier_discriminator_shared_fn,
                 classifier_private_fn,
                 discriminator_private_fn,
                 adversarial_loss_fn,
                 **hparams):

        # TODO: How to structure inputs? One x with guaranteed at least one attribute label and one without?

        # TODO: Need special input fn here with images without any of considered attributes as well?
        # Enough to just remove the filter?

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
            n_classes=n_classes,
            is_training=is_training,
            **hparams)

        # Model parts.
        encoder = tf.make_template("encoder", encoder_fn, is_training=is_training, **hparams)
        bottleneck = tf.make_template("bottleneck", bottleneck_fn, is_training=is_training, **hparams)
        decoder = tf.make_template("decoder", decoder_fn, is_training=is_training, **hparams)
        classifier = lambda x: _c_private(_cd_shared(x))
        discriminator = lambda x: _d_private(_cd_shared(x))

        # TODO: Does this model care about any middle parts? Like results of encode()
        def translate(x):
            z = encoder(x)
            b = bottleneck(z)
            translations = decoder(b, x, n_attributes)
            return translations

        x = preprocess(img)
        x_translated = translate(x)

        # TODO: For each translation?
        #d_adversarial_loss, g_adversarial_loss = adversarial_loss_fn(x, )

        # Reconstruction loss.
        # TODO: Should reconstruction loss be combined from all reconstruction combinations?

        disc_cls_variables = tf.trainable_variables("(classifier|discriminator)")
        enc_dec_variables = tf.trainable_variables("(encoder|bottleneck|decoder)")
        assert (set(disc_cls_variables) & set(enc_dec_variables)) == set(), "D and G should not share variables."

        global_step = tf.train.get_or_create_global_step()

        # TODO: Create update steps.

        scalar_summaries = tf.summary.merge((

        ))

        image_summaries = tf.summary.merge((

        ))