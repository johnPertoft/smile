import numpy as np
import skimage.io
import tensorflow as tf

from smile.experiments.summaries import img_summary
from smile.models import Model


# Note: DiscoGAN and CycleGAN are essentially the same with only minor differences mainly in architecture.


class DiscoGAN(Model):
    def __init__(self,
                 a_train, a_test, a_test_static,
                 b_train, b_test, b_test_static,
                 generator_fn,
                 discriminator_fn,
                 adversarial_loss_fn,
                 **hparams):

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        is_training = tf.placeholder_with_default(False, [])

        discriminator_a = tf.make_template("discriminator_a", discriminator_fn, is_training=is_training, **hparams)
        discriminator_b = tf.make_template("discriminator_b", discriminator_fn, is_training=is_training, **hparams)
        generator_ab = tf.make_template("generator_ab", generator_fn, is_training=is_training, **hparams)
        generator_ba = tf.make_template("generator_ba", generator_fn, is_training=is_training, **hparams)

        a = preprocess(a_train)
        b = preprocess(b_train)
        ba_translated = generator_ba(b)
        ab_translated = generator_ab(a)

        # TODO: Official code uses mse. Not mentioned in paper.
        reconstruction_loss_fn = tf.losses.absolute_difference

        # Loss parts.
        d_a_loss, g_ba_adv_loss = adversarial_loss_fn(a, ba_translated, discriminator_a, **hparams)
        d_b_loss, g_ab_adv_loss = adversarial_loss_fn(b, ab_translated, discriminator_b, **hparams)
        aba_reconstruction_loss = reconstruction_loss_fn(a, generator_ba(ab_translated))
        bab_reconstruction_loss = reconstruction_loss_fn(b, generator_ab(ba_translated))

        # TODO: Official code uses weights on loss parts. Not mentioned in paper.

        # Full objectives for generators.
        g_ab_loss = g_ab_adv_loss + aba_reconstruction_loss
        g_ba_loss = g_ba_adv_loss + bab_reconstruction_loss
        g_loss = g_ab_loss + g_ba_loss

        # Full objective for discriminators.
        d_loss = d_a_loss + d_b_loss

        def create_update_step(loss, variables):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                loss = loss + tf.losses.get_regularization_loss()
                return tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(loss, var_list=variables)

        global_step = tf.train.get_or_create_global_step()
        d_update_step = create_update_step(d_loss, tf.trainable_variables("discriminator"))
        g_update_step = create_update_step(g_loss, tf.trainable_variables("generator"))

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/g_ab", g_ab_loss),
            tf.summary.scalar("loss/g_ba", g_ba_loss),
            tf.summary.scalar("loss/rec_aba", aba_reconstruction_loss),
            tf.summary.scalar("loss/rec_bab", bab_reconstruction_loss),
            tf.summary.scalar("loss/adv_ab", g_ab_adv_loss),
            tf.summary.scalar("loss/adv_ba", g_ba_adv_loss),
            tf.summary.scalar("loss/d_a", d_a_loss),
            tf.summary.scalar("loss/d_b", d_b_loss),
            tf.summary.scalar("disc/a_real", tf.reduce_mean(discriminator_a(a))),
            tf.summary.scalar("disc/a_fake", tf.reduce_mean(discriminator_a(ba_translated))),
            tf.summary.scalar("disc/b_real", tf.reduce_mean(discriminator_b(b))),
            tf.summary.scalar("disc/b_fake", tf.reduce_mean(discriminator_b(ab_translated))),
        ))

        image_summaries = tf.summary.merge((
            img_summary("a_to_b_train", a_train, postprocess(ab_translated)),
            img_summary("b_to_a_train", b_train, postprocess(ba_translated)),
            img_summary("a_to_b_test", a_test, postprocess(generator_ab(preprocess(a_test)))),
            img_summary("b_to_a_test", b_test, postprocess(generator_ba(preprocess(b_test))))
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
        self.ba_translated_sample = tf.concat((
            b_test_static,
            postprocess(generator_ba(preprocess(b_test_static)))),
            axis=2)
        self.ab_translated_sample = tf.concat((
            a_test_static,
            postprocess(generator_ab(preprocess(a_test_static)))),
            axis=2)

        # Handles for exporting.
        self.a_input = tf.placeholder(tf.float32, [None] + a_train.get_shape().as_list()[1:])
        self.b_translated = postprocess(generator_ab(preprocess(self.a_input)))
        self.b_input = tf.placeholder(tf.float32, [None] + b_train.get_shape().as_list()[1:])
        self.a_translated = postprocess(generator_ba(preprocess(self.b_input)))

    def train_step(self, sess, summary_writer):
        for _ in range(self.n_discriminator_iters):
            sess.run(self.d_update_step, feed_dict={self.is_training: True})

        _, i = sess.run((self.g_update_step, self.global_step_increment), feed_dict={self.is_training: True})

        scalar_summaries = sess.run(self.scalar_summaries)
        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

        return i

    def generate_samples(self, sess, fname):
        ba, ab = sess.run((self.ba_translated_sample, self.ab_translated_sample))
        img = np.vstack(np.concatenate((ab, ba), axis=2))
        skimage.io.imsave(fname, img)

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
                "translate_ab": translation_signature(self.a_input, self.b_translated),
                "translate_ba": translation_signature(self.b_input, self.a_translated)
            })

        return builder.save()
