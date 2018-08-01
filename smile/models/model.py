import abc

import tensorflow as tf


class Model:
    @abc.abstractmethod
    def train_step(self, sess: tf.Session, summary_writer: tf.summary.FileWriter):
        pass

    @abc.abstractmethod
    def generate_samples(self, sess: tf.Session):
        pass
