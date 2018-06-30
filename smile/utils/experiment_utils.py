import argparse
import datetime
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf


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


def run_training(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    # get a model

    summary_writer = tf.summary.FileWriter(str(model_dir))

    scaffold = tf.train.Scaffold(local_init_op=tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        iterator_initializer))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
