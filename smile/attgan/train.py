from pathlib import Path
from typing import List

import tensorflow as tf

from smile.attgan import AttGAN
from smile.attgan.architectures import celeb
from smile.attgan.input import celeb_input_fn
from smile import utils

tf.logging.set_verbosity(tf.logging.INFO)


def run_training(model_dir: Path,
                 X_train_paths: List[Path],
                 X_test_paths: List[Path],
                 Y_train_paths: List[Path],
                 Y_test_paths: List[Path],
                 **hparams):

    pass


if __name__ == "__main__":
    arg_parser = utils.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
    arg_parser.add_argument("--X-train", nargs="+", required=True, help="Tfrecord train files for first image domain.")
    arg_parser.add_argument("--X-test", nargs="+", required=True, help="Tfrecord test files for first image domain.")
    arg_parser.add_argument("--Y-train", nargs="+", required=True, help="Tfrecord train files for second image domain.")
    arg_parser.add_argument("--Y-test", nargs="+", required=True, help="Tfrecord test files for second image domain.")

    arg_parser.add_hparam("batch-size", default=16, type=int, help="Batch size.")
    arg_parser.add_hparam("generator-architecture", default="paper", help="Architecture for generator network.")
    arg_parser.add_hparam("discriminator-architecture", default="paper", help="Architecture for discriminator network.")
    arg_parser.add_hparam("lambda-cyclic", default=5.0, type=float, help="Cyclic consistency loss weight.")
    arg_parser.add_hparam("use-history", action="store_true",
                          help="Whether a history of generated images should be shown to the discriminator.")
    arg_parser.add_hparam("growth-rate", default=16, type=int, help="Growth rate for densenet architecture.")

    args, hparams = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        model_dir = ROOT_RUNS_DIR / Path(utils.experiment_name("attgan", hparams))
    else:
        model_dir = Path(args.model_dir)

    run_training(
        model_dir,
        args.X_train,
        args.X_test,
        args.Y_train,
        args.Y_test,
        **hparams)