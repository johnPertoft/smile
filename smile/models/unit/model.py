import tensorflow as tf

from smile.experiments.summaries import img_summary
from smile.models import Model
from smile.models.unit.loss import cyclic_loss
from smile.models.unit.loss import vae_loss


class UNIT(Model):
    def __init__(self,
                 a_train, a_test, a_test_static,
                 b_train, b_test, b_test_static,
                 private_encoder_fn, shared_encoder_fn,
                 shared_decoder_fn, private_decoder_fn,
                 discriminator_fn,
                 adversarial_loss_fn,
                 **hparams):

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        # TODO: Remove this default value. Specify on each invocation instead?
        is_training = tf.placeholder_with_default(False, [])

        _encoder_a_private = tf.make_template(
            "encoder_a_private",
            private_encoder_fn,
            is_training=is_training,
            **hparams)
        _encoder_b_private = tf.make_template(
            "encoder_b_private",
            private_encoder_fn,
            is_training=is_training,
            **hparams)
        _encoder_shared = tf.make_template(
            "encoder_shared",
            shared_encoder_fn,
            is_training=is_training,
            **hparams)
        _decoder_shared = tf.make_template(
            "decoder_shared",
            shared_decoder_fn,
            is_training=is_training,
            **hparams)
        _decoder_a_private = tf.make_template(
            "decoder_a_private",
            private_decoder_fn,
            is_training=is_training,
            **hparams)
        _decoder_b_private = tf.make_template(
            "decoder_b_private",
            private_decoder_fn,
            is_training=is_training,
            **hparams)

        encoder_a = lambda x: _encoder_shared(_encoder_a_private(x))
        encoder_b = lambda x: _encoder_shared(_encoder_b_private(x))
        decoder_a = lambda z: _decoder_a_private(_decoder_shared(z))
        decoder_b = lambda z: _decoder_b_private(_decoder_shared(z))
        discriminator_a = tf.make_template("discriminator_a", discriminator_fn, is_training=is_training, **hparams)
        discriminator_b = tf.make_template("discriminator_b", discriminator_fn, is_training=is_training, **hparams)

        def sample_z(z_mu):
            return z_mu + tf.random_normal(tf.shape(z_mu), stddev=1.0)

        def translate_ab(a):
            return decoder_b(sample_z(encoder_a(a)))

        def translate_ba(b):
            return decoder_a(sample_z(encoder_b(b)))

        a = preprocess(a_train)
        b = preprocess(b_train)
        ab_translated = translate_ab(a)
        ba_translated = translate_ba(b)

        # TODO: Ok to use different z samples for different parts of the loss?
            # Closer to loss functions as defined in paper.
            # Cleaner implementation.
            # Downsides?

        # Loss parts.
        vae_a_loss = vae_loss(a, encoder_a, decoder_a, sample_z, **hparams)
        vae_b_loss = vae_loss(b, encoder_b, decoder_b, sample_z, **hparams)
        d_a_loss, g_a_loss = adversarial_loss_fn(a, ba_translated, discriminator_a, **hparams)
        d_b_loss, g_b_loss = adversarial_loss_fn(b, ab_translated, discriminator_b, **hparams)
        aba_cyclic_loss = cyclic_loss(a, encoder_a, decoder_a, encoder_b, decoder_b, sample_z, **hparams)
        bab_cyclic_loss = cyclic_loss(b, encoder_b, decoder_b, encoder_a, decoder_a, sample_z, **hparams)

        # Full generative objective.
        enc_dec_loss = hparams["lambda_adv"] * (g_a_loss + g_b_loss) + \
                       vae_a_loss + vae_b_loss + \
                       aba_cyclic_loss + bab_cyclic_loss

        learning_rate = 1e-4

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        d_update_step = tf.group(
            create_update_step(d_a_loss, tf.trainable_variables("discriminator_a")),
            create_update_step(d_b_loss, tf.trainable_variables("discriminator_b")))
        enc_dec_update_step = create_update_step(enc_dec_loss, tf.trainable_variables("(encoder|decoder)"))

        global_step = tf.train.get_or_create_global_step()

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/d_a_loss", d_a_loss),
            tf.summary.scalar("loss/d_b_loss", d_b_loss),
            tf.summary.scalar("loss/enc_dec_loss", enc_dec_loss),
            tf.summary.scalar("loss/parts/vae_a", vae_a_loss),  # TODO: Show kl vs reconstruction losses?
            tf.summary.scalar("loss/parts/vae_b", vae_b_loss),
            tf.summary.scalar("loss/parts/g_a", g_a_loss),
            tf.summary.scalar("loss/parts/g_b", g_b_loss),
            tf.summary.scalar("loss/parts/aba_cyclic", aba_cyclic_loss),
            tf.summary.scalar("loss/parts/bab_cyclic", bab_cyclic_loss),

            tf.summary.scalar("disc/a_real", tf.reduce_mean(discriminator_a(a))),
            tf.summary.scalar("disc/a_fake", tf.reduce_mean(discriminator_a(ba_translated))),
            tf.summary.scalar("disc/b_real", tf.reduce_mean(discriminator_b(b))),
            tf.summary.scalar("disc/b_fake", tf.reduce_mean(discriminator_b(ab_translated)))
        ))

        image_summaries = tf.summary.merge((
            img_summary("a_to_b_train", a_train, postprocess(ab_translated)),
            img_summary("b_to_a_train", b_train, postprocess(ba_translated)),
            img_summary("a_to_b_test", a_test, postprocess(translate_ab(preprocess(a_test)))),
            img_summary("b_to_a_test", b_test, postprocess(translate_ba(preprocess(b_test))))
        ))

        # Handles for training.
        self.is_training = is_training
        self.global_step_increment = global_step.assign_add(1)
        self.d_update_step = d_update_step
        self.enc_dec_update_step = enc_dec_update_step
        self.n_discriminator_iters = hparams["n_discriminator_iters"]
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

        # Handles for progress samples.
        self.ba_translated_sample = tf.concat((
            b_test_static,
            postprocess(translate_ba(preprocess(b_test_static)))),
            axis=2)
        self.ab_translated_sample = tf.concat((
            a_test_static,
            postprocess(translate_ab(preprocess(a_test_static)))),
            axis=2)

        # TODO: Are placeholders needed?
        # Handles for exporting.
        self.a_input = tf.placeholder(tf.float32, [None] + a_train.get_shape().as_list()[1:])
        self.ab_translated = postprocess(translate_ab(preprocess(self.a_input)))
        self.b_input = tf.placeholder(tf.float32, [None] + b_train.get_shape().as_list()[1:])
        self.ba_translated = postprocess(translate_ba(preprocess(self.b_input)))

    def train_step(self, sess, summary_writer):
        for _ in range(self.n_discriminator_iters):
            sess.run(self.d_update_step, feed_dict={self.is_training: True})

        _, scalar_summaries, i = sess.run(
            (self.enc_dec_update_step, self.scalar_summaries, self.global_step_increment),
            feed_dict={self.is_training: True})

        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def export(self, sess, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        def translation_signature(input_img, translated_img):
            return tf.saved_model.signature_def_utils.build_signature_def(
                inputs={"input": tf.saved_model.utils.build_tensor_info(input_img)},
                outputs={"translation": tf.saved_model.utils.build_tensor_info(translated_img)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "translate_ab": translation_signature(self.a_input, self.ab_translated),
                "translate_ba": translation_signature(self.b_input, self.ba_translated)
            })

        return builder.save()
