import tensorflow as tf


# TODO: attgan + self attention and spectral normalization

class SelfAttentionAttGAN:
    def __init__(self,
                 attribute_names,
                 img, attributes,
                 img_test, attributes_test):

        is_training = tf.placeholder_with_default(False, [])
        _, n_classes = attributes.get_shape()

    def train_step(self):
        pass
