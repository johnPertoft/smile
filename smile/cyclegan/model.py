import tensorflow as tf

from smile.cyclegan.loss import lsgan_losses


def preprocess(x):
    """[0, 1] -> [-1, 1]"""
    return x * 2 - 1


def postprocess(x):
    """[-1, 1] -> [0, 1]"""
    return (x + 1) / 2


class CycleGAN:
    def __init__(self, A, B, generator_fn, discriminator_fn, lambda_cyclic):
        is_training = tf.placeholder_with_default(False, [])

        A = preprocess(A)
        B = preprocess(B)

        # TODO: Solve dimension problems in another way. Maybe just resize it.
        # Skipping top and bottom rows as well as left and rightmost columns of each image.
        A = A[:, 1:-1, 1:-1, :]
        B = B[:, 1:-1, 1:-1, :]

        discriminator_a = tf.make_template("discriminator_A", discriminator_fn, is_training=is_training)
        discriminator_b = tf.make_template("discriminator_B", discriminator_fn, is_training=is_training)
        generator_ab = tf.make_template("generator_AB", generator_fn, is_training=is_training)
        generator_ba = tf.make_template("generator_BA", generator_fn, is_training=is_training)

        # Translations.
        A_generated = generator_ba(B)
        B_generated = generator_ab(A)

        global_step = tf.train.get_or_create_global_step()

        # TODO: Read paper again, need to have a buffer of history of some of the generated/translated images.
        # However, it wasn't clear which ones to actually keep. They used a buffer size of 50 I think. Does this
        # mean one from each of 50 update steps back or something else?
        with tf.variable_scope("history"):
            buffer_size = 50
            history_shape = [buffer_size] + A_generated.shape.as_list()[1:]
            generated_history_A = tf.get_variable(name="A", initializer=tf.zeros(history_shape, tf.float32))
            generated_history_B = tf.get_variable(name="B", initializer=tf.zeros(history_shape, tf.float32))
            current_index = global_step % buffer_size
            update_history = tf.group(
                generated_history_A[current_index].assign(A_generated[0]),
                generated_history_B[current_index].assign(B_generated[0]))

        # Adversarial loss (lsgan loss).
        D_A_real = discriminator_a(A)
        D_B_real = discriminator_b(B)
        D_A_fake = discriminator_a(A_generated)
        D_B_fake = discriminator_b(B_generated)
        D_A_loss, G_BA_adv_loss = lsgan_losses(D_A_real, D_A_fake)
        D_B_loss, G_AB_adv_loss = lsgan_losses(D_B_real, D_B_fake)

        # Cyclic consistency loss.
        B_reconstructed = generator_ab(A_generated)
        A_reconstructed = generator_ba(B_generated)
        ABA_cyclic_loss = tf.reduce_mean(tf.abs(A_reconstructed - A))
        BAB_cyclic_loss = tf.reduce_mean(tf.abs(B_reconstructed - B))
        cyclic_loss = lambda_cyclic * (ABA_cyclic_loss + BAB_cyclic_loss)

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
        # Just divide D_A_loss and D_B_loss?

        def get_vars(scope):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        def create_update_step(loss, variables):
            return tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=variables)

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

        image_summaries = tf.summary.merge((
            tf.summary.image("A_to_B", postprocess(tf.concat((A[:3], B_generated[:3]), axis=2))),
            tf.summary.image("B_to_A", postprocess(tf.concat((B[:3], A_generated[:3]), axis=2)))
        ))

        # TODO: possibly run on separate batches instead.
        train_op = tf.group(D_A_optimization_step,
                            G_AB_optimization_step,
                            D_B_optimization_step,
                            G_BA_optimization_step,
                            global_step.assign_add(1),
                            update_history)

        self.is_training = is_training
        self.A_generated = A_generated
        self.B_generated = B_generated
        self.train_op = train_op
        self.global_step = global_step
        self.scalar_summaries = scalar_summaries
        self.image_summaries = image_summaries

    def train_step(self, sess, summary_writer):
        feed_dict = {
            self.is_training: True
        }

        _, scalar_summaries, i = sess.run((self.train_op, self.scalar_summaries, self.global_step), feed_dict=feed_dict)

        summary_writer.add_summary(scalar_summaries, i)

        if i > 0 and i % 1000 == 0:
            image_summaries = sess.run(self.image_summaries)
            summary_writer.add_summary(image_summaries, i)

    def export(self):
        pass
