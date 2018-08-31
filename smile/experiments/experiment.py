import argparse
import datetime
from pathlib import Path
from typing import Any
from typing import Dict

import imageio
import tensorflow as tf

from smile.models import Model


ROOT_RUNS_DIR = Path("runs")


def experiment_name(modelname: str, hparams: Dict[str, Any]):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    hparams_string = "_".join(f"{k}={hparams[k]}" for k in sorted(hparams.keys()))
    return f"{modelname}_{timestamp}_{hparams_string}"


class ArgumentParser(argparse.ArgumentParser):
    """Adds a method on top of `ArgumentParser` to separate hparams from other input args."""

    def add_hparam(self, argument, *args, **kwargs):
        hparam_key = argument
        if argument.startswith("--"):
            hparam_key = argument[2:]
        else:
            argument = f"--{argument}"
        hparam_key = hparam_key.replace("-", "_")

        if not hasattr(self, '_hparam_keys'):
            self._hparam_keys = set()
        self._hparam_keys.add(hparam_key)

        argparse.ArgumentParser.add_argument(self, f"{argument}", *args, **kwargs)

    def parse_args(self):
        args = argparse.ArgumentParser.parse_args(self)

        hparams = {}
        if hasattr(self, '_hparam_keys'):
            hparams = {k: args.__dict__[k] for k in self._hparam_keys}
            for k in self._hparam_keys:
                args.__dict__.pop(k)

        return args, hparams


def run_experiment(model_dir: Path,
                   model: Model,
                   n_training_step: int,
                   custom_init_op: tf.Operation=None,
                   **hparams):

    model_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.FileWriter(str(model_dir))
    sample_frequency = 10000

    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer(),
                tf.tables_initializer()]
    if custom_init_op is not None:
        init_ops.append(custom_init_op)

    scaffold = tf.train.Scaffold(local_init_op=tf.group(init_ops))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            config=config,
            checkpoint_dir=str(model_dir),
            save_summaries_secs=30) as sess:

        while not sess.should_stop():
            i = model.train_step(sess, summary_writer)

            # TODO: Specify num epochs instead? Refactor input fns to return dataset instead.
            # TODO: Handle summary writes at different frequencies too. Handle summary writes here?

            if i > 0 and i % sample_frequency == 0:
                model.generate_samples(sess, str(model_dir / f"testsamples_{i}.png"))

            if i > n_training_step:
                break

        model.generate_samples(sess, str(model_dir / "testsamples_final.png"))

    # TODO: Generate a gif of testsamples.

    # Note: tf.train.MonitoredTrainingSession finalizes the graph so can't export from it.
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(str(model_dir)))
        model.export(sess, str(model_dir / "export"))

    # TODO: One test set input for summaries
    # TODO: One very small set of test images for generating over time
    #   * Should be the same over the whole training.
    #   * Should be the same for all models
    #   * For binary translation models and non binary this might be difficult.
    # model.generate_samples should accept input tensors.
    # TODO: Add script for easily generating more samples given a saved_model.

    # TODO: Add standardized implementations of the following for easier experimentation
    #   * Losses
    #   * Architectures
    #   * Regularization (e.g. gradient penalty)
    #   * Normalization (e.g. spectral normalization)
    #   * Etc, like self-attention layers.

