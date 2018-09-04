import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw


def img_summary(name, left, right):
    side_by_side = tf.concat((left, right), axis=2)
    return tf.summary.image(name, side_by_side)


def img_summary_with_text(name, attribute_names,
                          left_imgs, left_attributes_indicator,
                          right_imgs, right_attributes_indicator):

    text_row_vertical_space = 20
    extra_vertical_space = text_row_vertical_space * len(attribute_names)
    h = int(left_imgs.shape[1])

    def put_text(imgs, attrs):
        imgs_with_text = []
        for img, attr in zip(imgs, attrs):
            active_attributes = [an for an, a in zip(attribute_names, attr) if a == 1]
            img = Image.fromarray(np.uint8(img * 255))
            draw = ImageDraw.Draw(img)
            draw.text((0, h), "\n".join(active_attributes), (0, 0, 0))
            imgs_with_text.append(np.array(img) / 255.0)
        return np.stack(imgs_with_text).astype(np.float32)

    def add_attributes_text(x, attr):
        x = tf.pad(x, [[0, 0], [0, extra_vertical_space], [0, 0], [0, 0]], constant_values=1.0)
        x = x[:3]
        x = tf.py_func(put_text, [x, attr], tf.float32)
        return x

    left_imgs = add_attributes_text(left_imgs, left_attributes_indicator)
    right_imgs = add_attributes_text(right_imgs, right_attributes_indicator)

    return img_summary(name, left_imgs, right_imgs)