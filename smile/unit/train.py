import tensorflow as tf


# temp
A = tf.placeholder(shape=(None, 218, 178, 3), dtype=tf.float32)
B = tf.placeholder(shape=(None, 218, 178, 3), dtype=tf.float32)

from .model import UNIT
from .architectures.celeb import paper

hparams = {
    "lambda_vae_kl": 0.1,
    "lambda_vae_nll": 100.0,
    "lambda_gan": 10.0,
    "lambda_cyclic_kl": 0.1,
    "lambda_cyclic_nll": 100.0
}

model = UNIT(A, A,
             B, B,
             paper.private_encoder, paper.shared_encoder,
             paper.shared_decoder, paper.private_decoder,
             paper.discriminator,
             **hparams)
