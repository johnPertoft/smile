import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw


def img_summary(name, before, after):
    # TODO: Optional text on image somewhere.

    side_by_side = tf.concat((before, after), axis=2)
    return tf.summary.image(name, side_by_side, max_outputs=3)


# TODO: get numpy image, return with text attached somehow
# TODO: tf.py_func

def _put_text(img, text, pos):
    # fix number range to [0, 1] first
    img = Image.fromarray(np.uint8(img * 255))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, (255, 255, 255))
    return np.array(img)


def img_summary_with_text(name, left_imgs, left_text, right_imgs, right_text):

    # Create space for the full image

    side_by_side = tf.concat((left_imgs, right_imgs), axis=2)

    # Determine how much space is needed for text.

    # Pad to make space for text

    # One line per attribute?

    extra_vertical_space = 100

    img = tf.pad(side_by_side, [[0, 0], [0, extra_vertical_space], [0, 0], [0, 0]], constant_values=1.0)

    # Wrap the function to add the text as a tf.py_func
    # tf.map_fn(tf.py_func(_put_text

    img = tf.py_func(_put_text, )

    return tf.summary.image(name, img, max_outputs=3)
