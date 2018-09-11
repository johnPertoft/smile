import tensorflow as tf


def spectral_normalization(W, n_power_iterations):
    # TODO: Remove the transposes by reformulating equations a bit.

    original_kernel_shape = W.shape
    d_out = original_kernel_shape[-1]

    W_sn = W
    W_sn = tf.reshape(W_sn, (-1, d_out))
    W_sn = tf.transpose(W_sn)

    u = tf.get_variable(
        "u",
        shape=(d_out, 1),
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
        trainable=False)

    def power_iteration_step(i, u_i, _v_i):
        v_i = tf.nn.l2_normalize(tf.transpose(W_sn) @ u_i)
        u_i = tf.nn.l2_normalize(W_sn @ v_i)
        return i + 1, u_i, v_i

    _, u_pi, v_pi = tf.while_loop(
        cond=lambda i, _u, _v: i < n_power_iterations,
        body=power_iteration_step,
        loop_vars=(
            tf.constant(0, tf.int32),
            u,
            tf.zeros((W_sn.shape[1], 1))))

    u_pi = tf.stop_gradient(u_pi)
    v_pi = tf.stop_gradient(v_pi)

    # TODO: How should this be handled at test time?
    # I suppose we should keep a running average of sigma which is
    # used as a constant after training. Like batch norm.

    # Spectral-normalized weights.
    W_sn = W_sn / (tf.transpose(u_pi) @ W_sn @ v_pi)
    W_sn = tf.transpose(W_sn)
    W_sn = tf.reshape(W_sn, original_kernel_shape)

    # Update op for u.
    update_u = u.assign(u_pi)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_u)

    return W_sn
