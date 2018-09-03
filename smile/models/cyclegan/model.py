import numpy as np
import skimage.io
import tensorflow as tf

from smile.losses import lsgan_losses
from smile.models import Model


class CycleGAN(Model):
    def __init__(self,
                 a_train, a_test, a_test_static,
                 b_train, b_test, b_test_static,
                 generator_fn, discriminator_fn,
                 **hparams):

        def preprocess(x):
            return x * 2 - 1

        def postprocess(x):
            return (x + 1) / 2

        a = preprocess(a_train)
        b = preprocess(b_train)
        is_training = tf.placeholder_with_default(False, [])

        discriminator_a = tf.make_template("discriminator_a", discriminator_fn, is_training=is_training, **hparams)
        discriminator_b = tf.make_template("discriminator_b", discriminator_fn, is_training=is_training, **hparams)
        generator_ab = tf.make_template("generator_ab", generator_fn, is_training=is_training, **hparams)
        generator_ba = tf.make_template("generator_ba", generator_fn, is_training=is_training, **hparams)

        # Translations.
        a_translated = generator_ba(b)
        b_translated = generator_ab(a)

        global_step = tf.train.get_or_create_global_step()

        # TODO: Read paper again. Rewrite implementation. tf.losses.add_loss instead + tf.losses.get_losses
        # I think paper used batch size 1, but we can just pick the first image per batch for higher batch sizes
        # I guess.
        update_history = None
        if hparams["use_history"]:
            with tf.variable_scope("history"):
                # TODO: tfgan implementation randomly samples from history and current. Try this?
                buffer_size = 50
                history_shape = [buffer_size] + a_translated.shape.as_list()[1:]
                generated_history_a = tf.get_variable(name="a", initializer=tf.zeros(history_shape, tf.float32))
                generated_history_b = tf.get_variable(name="b", initializer=tf.zeros(history_shape, tf.float32))
                current_index = global_step % buffer_size
                update_history = tf.group(
                    generated_history_a[current_index].assign(a_translated[0]),
                    generated_history_b[current_index].assign(b_translated[0]))

        # Adversarial loss (lsgan loss).
        d_a_loss, g_ba_adv_loss = lsgan_losses(a, a_translated, discriminator_a)
        d_b_loss, g_ab_adv_loss = lsgan_losses(b, b_translated, discriminator_b)

        # Cyclic consistency loss.
        b_reconstructed = generator_ab(a_translated)
        a_reconstructed = generator_ba(b_translated)
        aba_cyclic_loss = tf.reduce_mean(tf.abs(a_reconstructed - a))
        bab_cyclic_loss = tf.reduce_mean(tf.abs(b_reconstructed - b))
        cyclic_loss = hparams["lambda_cyclic"] * (aba_cyclic_loss + bab_cyclic_loss)

        # Combined loss for generators.
        g_ab_loss = g_ab_adv_loss + cyclic_loss
        g_ba_loss = g_ba_adv_loss + cyclic_loss

        initial_learning_rate = 2e-4
        start_decay_step = 100 * 5_000  # Rough estimate of 100 epochs.
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[start_decay_step],
                                                    values=[initial_learning_rate, initial_learning_rate / 5])

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        # From paper: "In practice, we divide the objective by 2 while optimizing D".
        d_a_optimization_step = create_update_step(d_a_loss * 0.5, get_vars("discriminator_a"))
        d_b_optimization_step = create_update_step(d_b_loss * 0.5, get_vars("discriminator_b"))

        g_ab_optimization_step = create_update_step(g_ab_loss, get_vars("generator_ab"))
        g_ba_optimization_step = create_update_step(g_ba_loss, get_vars("generator_ba"))

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
            tf.summary.scalar("disc_a/fake", tf.reduce_mean(discriminator_a(a_translated))),
            tf.summary.scalar("disc_b/real", tf.reduce_mean(discriminator_b(b))),
            tf.summary.scalar("disc_b/fake", tf.reduce_mean(discriminator_b(b_translated))),
            tf.summary.scalar("learning_rate", learning_rate)
        ))

        def side_by_side_summary(name, img1, img2):
            img1 = postprocess(img1[:3])
            img2 = postprocess(img2[:3])
            return tf.summary.image(name, tf.concat((img1, img2), axis=2))

        image_summaries = tf.summary.merge((
            side_by_side_summary("a_to_b_train", postprocess(a), postprocess(b_translated)),
            side_by_side_summary("b_to_a_train", postprocess(b), postprocess(a_translated)),
            side_by_side_summary("a_to_b_test", a_test, postprocess(generator_ab(preprocess(a_test)))),
            side_by_side_summary("b_to_a_test", b_test, postprocess(generator_ba(preprocess(b_test))))
        ))

        train_step_ops = [d_a_optimization_step, d_b_optimization_step,
                          g_ab_optimization_step, g_ba_optimization_step,
                          global_step.assign_add(1)]
        if update_history is not None:
            train_step_ops.append(update_history)
        train_op = tf.group(*train_step_ops)

        # Handles for training.
        self.is_training = is_training
        self.train_op = train_op
        self.global_step = global_step
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
        _, scalar_summaries, i = sess.run(
            (self.train_op, self.scalar_summaries, self.global_step),
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
