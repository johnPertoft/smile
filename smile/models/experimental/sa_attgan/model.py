import tensorflow as tf

from smile.models import Model
from smile.models.experimental import networks


def preprocess(x):
    return x * 2 - 1


def postprocess(x):
    return (x + 1) / 2


# TODO: attgan + self attention and spectral normalization
# TODO: progressive growing.


class SelfAttentionAttGAN(Model):
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test,
                 **hparams):

        # TODO: Take adversarial loss fn as input.

        is_training = tf.placeholder_with_default(False, [])
        _, n_classes = attributes.get_shape()

        _cd_shared = tf.make_template(
            "classifier_discriminator_shared",
            networks.classifier_discriminator_shared,
            is_training=is_training,
            **hparams)
        _d_private = tf.make_template(
            "discriminator_private",
            networks.discriminator_private,
            is_training=is_training,
            **hparams)
        _c_private = tf.make_template(
            "classifier_private",
            networks.classifier_private,
            n_classes=n_classes,
            is_training=is_training,
            **hparams)

        # Model parts.
        encode = tf.make_template("encoder", networks.encoder, is_training=is_training, **hparams)
        decode = tf.make_template("decoder", networks.decoder, is_training=is_training, **hparams)
        translate = lambda x, attr: decode(encode(x), attr)
        classify = lambda x: _c_private(_cd_shared(x))
        discriminate = lambda x: _d_private(_cd_shared(x))

        def generate_attributes(attributes):
            # 50/50 sample per attribute.
            sampled_attributes = \
                tf.cast(tf.random_uniform(shape=tf.shape(attributes), dtype=tf.int32, maxval=2), tf.float32)
            return sampled_attributes

        x = preprocess(img)
        z = encode(x)
        sampled_attributes = generate_attributes(attributes)
        x_translated = translate(x, sampled_attributes)
        x_reconstructed = translate(x, attributes)


        exit()

    def train_step(self, sess, summary_writer):
        pass

    def generate_samples(self, sess: tf.Session):
        pass
