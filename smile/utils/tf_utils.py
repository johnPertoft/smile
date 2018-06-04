import tensorflow as tf


def img_summary(name, before, after):
    # TODO: Optional text on image somewhere.

    side_by_side = tf.concat((before, after), axis=2)
    return tf.summary.image(name, side_by_side, max_outputs=3)
