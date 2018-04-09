import tensorflow as tf


# temp
A = tf.placeholder(shape=(None, 218, 178, 3), dtype=tf.float32)
B = tf.placeholder(shape=(None, 218, 178, 3), dtype=tf.float32)

from .model import UNIT

model = UNIT(A, B)