import numpy as np
import skimage.io
import tensorflow as tf

from smile.experiments.summaries import img_summary
from smile.losses import lsgan_losses
from smile.models import Model


class CycleGAN(Model):
    def __init__(self,
                 a_train, a_test, a_test_static,
                 b_train, b_test, b_test_static,
                 generator_fn, discriminator_fn,
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

        global_step = tf.train.get_or_create_global_step()

        # TODO: Read paper again. Rewrite implementation of this. tf.losses.add_loss instead + tf.losses.get_losses
        # I think paper used batch size 1, but we can just pick the first image per batch for higher batch sizes
        # I guess.
        update_history = None
        if hparams["use_history"]:
            raise NotImplementedError("Test implementation.")
            with tf.variable_scope("history"):
                # TODO: tfgan implementation randomly samples from history and current. Try this?
                buffer_size = 50
                history_shape = [buffer_size] + ba_translated.shape.as_list()[1:]
                generated_history_a = tf.get_variable(name="a", initializer=tf.zeros(history_shape, tf.float32))
                generated_history_b = tf.get_variable(name="b", initializer=tf.zeros(history_shape, tf.float32))
                current_index = global_step % buffer_size
                update_history = tf.group(
                    generated_history_a[current_index].assign(ba_translated[0]),
                    generated_history_b[current_index].assign(ab_translated[0]))

        # Loss parts.
        d_a_loss, g_ba_adv_loss = adversarial_loss_fn(a, ba_translated, discriminator_a, **hparams)
        d_b_loss, g_ab_adv_loss = adversarial_loss_fn(b, ab_translated, discriminator_b, **hparams)
        aba_cyclic_loss = tf.losses.absolute_difference(a, generator_ba(ab_translated))
        bab_cyclic_loss = tf.losses.absolute_difference(b, generator_ab(ba_translated))
        cyclic_loss = hparams["lambda_cyclic"] * (aba_cyclic_loss + bab_cyclic_loss)

        # Full objectives for generators.
        g_ab_loss = g_ab_adv_loss + cyclic_loss
        g_ba_loss = g_ba_adv_loss + cyclic_loss

        initial_learning_rate = 2e-4
        start_decay_step = 100 * 5_000  # Rough estimate of 100 epochs.
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[start_decay_step],
                                                    values=[initial_learning_rate, initial_learning_rate / 5])

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        # From paper: "In practice, we divide the objective by 2 while optimizing D".
        d_a_update_step = create_update_step(d_a_loss * 0.5, tf.trainable_variables("discriminator_a"))
        d_b_update_step = create_update_step(d_b_loss * 0.5, tf.trainable_variables("discriminator_b"))
        d_update_step = tf.group(d_a_update_step, d_b_update_step)

        g_ab_update_step = create_update_step(g_ab_loss, tf.trainable_variables("generator_ab"))
        g_ba_update_step = create_update_step(g_ba_loss, tf.trainable_variables("generator_ba"))
        g_update_step = tf.group(g_ab_update_step, g_ba_update_step)

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/g_ab", g_ab_loss),
            tf.summary.scalar("loss/g_ba", g_ba_loss),
            tf.summary.scalar("loss/aba_cyclic", aba_cyclic_loss),
            tf.summary.scalar("loss/bab_cyclic", bab_cyclic_loss),
            tf.summary.scalar("loss/g_ab_adv", g_ab_adv_loss),
            tf.summary.scalar("loss/g_ba_adv", g_ba_adv_loss),
            tf.summary.scalar("loss/d_a", d_a_loss),
            tf.summary.scalar("loss/d_b", d_b_loss),
            tf.summary.scalar("disc_a/real", tf.reduce_mean(discriminator_a(a))),
            tf.summary.scalar("disc_a/fake", tf.reduce_mean(discriminator_a(ba_translated))),
            tf.summary.scalar("disc_b/real", tf.reduce_mean(discriminator_b(b))),
            tf.summary.scalar("disc_b/fake", tf.reduce_mean(discriminator_b(ab_translated))),
            tf.summary.scalar("learning_rate", learning_rate)
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

        # TODO: Are placeholders needed?
        # Handles for exporting.
        self.a_input = tf.placeholder(tf.float32, [None] + a_train.get_shape().as_list()[1:])
        self.b_translated = postprocess(generator_ab(preprocess(self.a_input)))
        self.b_input = tf.placeholder(tf.float32, [None] + b_train.get_shape().as_list()[1:])
        self.a_translated = postprocess(generator_ba(preprocess(self.b_input)))

    def train_step(self, sess, summary_writer):
        for _ in range(self.n_discriminator_iters):
            sess.run(self.d_update_step, feed_dict={self.is_training: True})

        _, scalar_summaries, i = sess.run(
            (self.g_update_step, self.scalar_summaries, self.global_step_increment),
            feed_dict={self.is_training: True})

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
