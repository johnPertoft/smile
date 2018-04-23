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

        # TODO: Define eta sampling somewhere.

        def translate_a_to_b(a):
            z_mu = encoder_a(a)
            return decoder_b(z_mu + tf.random_normal(tf.shape(z_mu), stddev=1.0))

        def translate_b_to_a(b):
            z_mu = encoder_b(b)
            return decoder_a(z_mu + tf.random_normal(tf.shape(z_mu), stddev=1.0))

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

        # TODO: Add summaries for parts of the losses here, i.e. vae_loss = kl + nll losses
        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/vae_loss_a", vae_loss_a),
            tf.summary.scalar("loss/vae_loss_b", vae_loss_b),
            tf.summary.scalar("loss/gen_loss_a", gen_loss_a),
            tf.summary.scalar("loss/gen_loss_b", gen_loss_b),
            tf.summary.scalar("loss/disc_loss_a", disc_loss_a),
            tf.summary.scalar("loss/disc_loss_b", disc_loss_b),
            tf.summary.scalar("loss/cyclic_loss_a", cyclic_loss_a),
            tf.summary.scalar("loss/cyclic_loss_b", cyclic_loss_b),
            tf.summary.scalar("disc_a/real", tf.reduce_mean(discriminator_a(A))),
            tf.summary.scalar("disc_a/fake", tf.reduce_mean(discriminator_a(a_translated))),
            tf.summary.scalar("disc_b/real", tf.reduce_mean(discriminator_b(B))),
            tf.summary.scalar("disc_b/fake", tf.reduce_mean(discriminator_b(b_translated))),
            tf.summary.scalar("learning_rate", learning_rate)
        ))

        # TODO: Move out image comparison summary elsewhere. Used for every model.
        a_translated_test = postprocess(translate_b_to_a(preprocess(B_test)))
        b_translated_test = postprocess(translate_a_to_b(preprocess(A_test)))
        image_summaries = tf.summary.merge((
            tf.summary.image("A_to_B_train", tf.concat((A_train[:3], postprocess(b_translated[:3])), axis=2)),
            tf.summary.image("B_to_A_train", tf.concat((B_train[:3], postprocess(a_translated[:3])), axis=2)),
            tf.summary.image("A_to_B_test", tf.concat((A_test[:3], b_translated_test[:3]), axis=2)),
            tf.summary.image("B_to_A_test", tf.concat((B_test[:3], a_translated_test[:3]), axis=2))
        ))

        self.a_translated = a_translated
        self.b_translated = b_translated
        self.is_training = is_training
        self.train_op = train_op
        self.global_step = global_step
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
