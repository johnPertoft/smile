import abc

import tensorflow as tf


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train_step(self, sess: tf.Session, summary_writer: tf.summary.FileWriter):
        pass

    @abc.abstractmethod
    def generate_samples(self, sess: tf.Session, fname: str):
        pass

    @abc.abstractmethod
    def export(self, sess: tf.Session, export_dir: str):
        # TODO: Put constants for saved_model signatures here.
        pass
