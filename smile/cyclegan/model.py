import tensorflow as tf

from smile.cyclegan.loss import lsgan_losses


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


class CycleGAN:
    def __init__(self,
                 A_train, A_test,
                 B_train, B_test,
                 generator_fn, discriminator_fn,
                 **hparams):

        A = A_train
        B = B_train
        is_training = tf.placeholder_with_default(False, [])

        A = preprocess(A)
        B = preprocess(B)

        discriminator_a = tf.make_template("discriminator_A", discriminator_fn, is_training=is_training, **hparams)
        discriminator_b = tf.make_template("discriminator_B", discriminator_fn, is_training=is_training, **hparams)
        generator_ab = tf.make_template("generator_AB", generator_fn, is_training=is_training, **hparams)
        generator_ba = tf.make_template("generator_BA", generator_fn, is_training=is_training, **hparams)

        # Translations.
        A_translated = generator_ba(B)
        B_translated = generator_ab(A)

        global_step = tf.train.get_or_create_global_step()

        # TODO: Read paper again.
        # I think paper used batch size 1, but we can just pick the first image per batch for higher batch sizes
        # I guess.
        update_history = None
        if hparams["use_history"]:
            with tf.variable_scope("history"):
                # TODO: tfgan implementation randomly samples from history and current. Try this?
                buffer_size = 50
                history_shape = [buffer_size] + A_translated.shape.as_list()[1:]
                generated_history_A = tf.get_variable(name="A", initializer=tf.zeros(history_shape, tf.float32))
                generated_history_B = tf.get_variable(name="B", initializer=tf.zeros(history_shape, tf.float32))
                current_index = global_step % buffer_size
                update_history = tf.group(
                    generated_history_A[current_index].assign(A_translated[0]),
                    generated_history_B[current_index].assign(B_translated[0]))

        # Adversarial loss (lsgan loss).
        D_A_real = discriminator_a(A)
        D_B_real = discriminator_b(B)
        D_A_fake = discriminator_a(A_translated)
        D_B_fake = discriminator_b(B_translated)
        D_A_loss, G_BA_adv_loss = lsgan_losses(D_A_real, D_A_fake)
        D_B_loss, G_AB_adv_loss = lsgan_losses(D_B_real, D_B_fake)

        # Cyclic consistency loss.
        B_reconstructed = generator_ab(A_translated)
        A_reconstructed = generator_ba(B_translated)
        ABA_cyclic_loss = tf.reduce_mean(tf.abs(A_reconstructed - A))
        BAB_cyclic_loss = tf.reduce_mean(tf.abs(B_reconstructed - B))
        cyclic_loss = hparams["lambda_cyclic"] * (ABA_cyclic_loss + BAB_cyclic_loss)

        # Combined loss for generators.
        G_AB_loss = G_AB_adv_loss + cyclic_loss
        G_BA_loss = G_BA_adv_loss + cyclic_loss

        initial_learning_rate = 2e-4
        start_decay_step = 100 * 5_000  # Rough estimate of 100 epochs.
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    boundaries=[start_decay_step],
                                                    values=[initial_learning_rate, initial_learning_rate / 5])
        """
        learning_rate = tf.cond(
            global_step < start_decay_step,
            lambda: initial_learning_rate,
            lambda: tf.train.polynomial_decay(
                learning_rate=initial_learning_rate,
                global_step=global_step - start_decay_step,
                decay_steps=100_000,
                end_learning_rate=0.0,
                power=1.0))
        """

        # TODO: from paper:
        # "In practice, we divide the objective by 2 while optimizing D"
        D_A_loss = D_A_loss * 0.5
        D_B_loss = D_B_loss * 0.5

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, var_list=variables)

        D_A_optimization_step = create_update_step(D_A_loss, get_vars("discriminator_A"))
        D_B_optimization_step = create_update_step(D_B_loss, get_vars("discriminator_B"))
        G_AB_optimization_step = create_update_step(G_AB_loss, get_vars("generator_AB"))
        G_BA_optimization_step = create_update_step(G_BA_loss, get_vars("generator_BA"))

        scalar_summaries = tf.summary.merge((
            tf.summary.scalar("loss/G_AB", G_AB_loss),
            tf.summary.scalar("loss/G_BA", G_BA_loss),
            tf.summary.scalar("loss/ABA_cyclic", ABA_cyclic_loss),
            tf.summary.scalar("loss/BAB_cyclic", BAB_cyclic_loss),
            tf.summary.scalar("loss/G_AB_adv", G_AB_adv_loss),
            tf.summary.scalar("loss/G_BA_adv", G_BA_adv_loss),
            tf.summary.scalar("loss/D_A", D_A_loss),
            tf.summary.scalar("loss/D_B", D_B_loss),
            tf.summary.scalar("disc_a/real", tf.reduce_mean(D_A_real)),
            tf.summary.scalar("disc_a/fake", tf.reduce_mean(D_A_fake)),
            tf.summary.scalar("disc_b/real", tf.reduce_mean(D_B_real)),
            tf.summary.scalar("disc_b/fake", tf.reduce_mean(D_B_fake)),
            tf.summary.scalar("learning_rate", learning_rate)
        ))

        # TODO: gradient summaries.
        # TODO: Show heatmap of discriminator output.
        # TODO: Show heatmap of what discriminator cares about? should be mainly mouth etc.

        A_translated_test = postprocess(generator_ba(preprocess(B_test)))
        B_translated_test = postprocess(generator_ab(preprocess(A_test)))

        image_summaries = tf.summary.merge((
            tf.summary.image("A_to_B_train", tf.concat((postprocess(A[:3]), postprocess(B_translated[:3])), axis=2)),
            tf.summary.image("B_to_A_train", tf.concat((postprocess(B[:3]), postprocess(A_translated[:3])), axis=2)),
            tf.summary.image("A_to_B_test", tf.concat((A_test[:3], B_translated_test[:3]), axis=2)),
            tf.summary.image("B_to_A_test", tf.concat((B_test[:3], A_translated_test[:3]), axis=2))
        ))

        # TODO: possibly run on separate batches instead.
        train_step_ops = [D_A_optimization_step, D_B_optimization_step,
                          G_AB_optimization_step, G_BA_optimization_step,
                          global_step.assign_add(1)]
        if update_history is not None:
            train_step_ops.append(update_history)
        train_op = tf.group(*train_step_ops)

        # Handles for training.
        self.is_training = is_training
        self.A_generated = A_translated
        self.B_generated = B_translated
        self.train_op = train_op
        self.global_step = global_step
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

        # Handles for exporting.
        self.A_input = tf.placeholder(tf.float32, [None] + A_train.get_shape().as_list()[1:])
        self.B_translated = postprocess(generator_ab(preprocess(self.A_input)))
        self.B_input = tf.placeholder(tf.float32, [None] + B_train.get_shape().as_list()[1:])
        self.A_translated = postprocess(generator_ba(preprocess(self.B_input)))

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
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        # TODO: Only include the necessary parts of the graph.

        def translation_signature(input_img, translated_img):
            return tf.saved_model.signature_def_utils.build_signature_def(
                inputs={"input": tf.saved_model.utils.build_tensor_info(input_img)},
                outputs={"translation": tf.saved_model.utils.build_tensor_info(translated_img)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "translate_ab": translation_signature(self.A_input, self.B_translated),
                "translate_ba": translation_signature(self.B_input, self.A_translated)
            })

        return builder.save()
