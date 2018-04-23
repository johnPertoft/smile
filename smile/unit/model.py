import tensorflow as tf

from smile.unit.loss import vae_loss, gan_losses, cyclic_loss


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


class UNIT:
    def __init__(self,
                 A_train, A_test,
                 B_train, B_test,
                 private_encoder_fn, shared_encoder_fn,
                 shared_decoder_fn, private_decoder_fn,
                 discriminator_fn,
                 **hparams):

        A = A_train
        B = B_train
        is_training = tf.placeholder_with_default(False, [])

        A = preprocess(A)
        B = preprocess(B)

        # Encoder subgraphs.
        encoder_a_private = tf.make_template(
            "encoder_a_private",
            private_encoder_fn,
            is_training=is_training,
            **hparams)
        encoder_b_private = tf.make_template(
            "encoder_b_private",
            private_encoder_fn,
            is_training=is_training,
            **hparams)
        encoder_shared = tf.make_template(
            "encoder_shared",
            shared_encoder_fn,
            is_training=is_training,
            **hparams)
        encoder_a = lambda x: encoder_shared(encoder_a_private(x))
        encoder_b = lambda x: encoder_shared(encoder_b_private(x))

        # Decoder subgraphs.
        decoder_shared = tf.make_template(
            "decoder_shared",
            shared_decoder_fn,
            is_training=is_training,
            **hparams)
        decoder_a_private = tf.make_template(
            "decoder_a_private",
            private_decoder_fn,
            is_training=is_training,
            **hparams)
        decoder_b_private = tf.make_template(
            "decoder_b_private",
            private_decoder_fn,
            is_training=is_training,
            **hparams)
        decoder_a = lambda z: decoder_a_private(decoder_shared(z))
        decoder_b = lambda z: decoder_b_private(decoder_shared(z))

        # TODO: Make functions for the different "streams".

        # Discriminator subgraphs.
        discriminator_a = tf.make_template("discriminator_a", discriminator_fn, is_training=is_training, **hparams)
        discriminator_b = tf.make_template("discriminator_b", discriminator_fn, is_training=is_training, **hparams)

        # Encoder computed posterior means.
        z_mu_a = encoder_a(A)
        z_mu_b = encoder_b(B)

        # Reparametrization for sampling of z from posterior.
        z_a = z_mu_a + tf.random_normal(tf.shape(z_mu_a), stddev=1.0)
        z_b = z_mu_b + tf.random_normal(tf.shape(z_mu_b), stddev=1.0)

        # Translations.
        a_translated = decoder_a(z_b)
        b_translated = decoder_b(z_a)

        # Losses.
        vae_loss_a = vae_loss(A, encoder_a, decoder_a, **hparams)
        vae_loss_b = vae_loss(B, encoder_b, decoder_b, **hparams)
        disc_loss_a, gen_loss_a = gan_losses(discriminator_a(A), discriminator_a(a_translated), **hparams)
        disc_loss_b, gen_loss_b = gan_losses(discriminator_b(B), discriminator_b(b_translated), **hparams)
        cyclic_loss_a = cyclic_loss(A, encoder_a, decoder_a, encoder_b, decoder_b, **hparams)
        cyclic_loss_b = cyclic_loss(B, encoder_b, decoder_b, encoder_a, decoder_b, **hparams)
        generative_loss = vae_loss_a + vae_loss_b + gen_loss_a + gen_loss_b + cyclic_loss_a + cyclic_loss_b

        learning_rate = 1e-4

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        global_step = tf.train.get_or_create_global_step()
        train_op = tf.group(
            create_update_step(disc_loss_a, get_vars("discriminator_a")),
            create_update_step(disc_loss_b, get_vars("discriminator_b")),
            create_update_step(generative_loss, get_vars("(encoder|decoder)")),
            global_step.assign_add(1))

        self.a_translated = a_translated
        self.b_translated = b_translated
        self.is_training = is_training
        self.train_op = train_op
        self.global_step = global_step

    def train_step(self, sess, summary_writer):
        pass
