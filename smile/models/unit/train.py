from pathlib import Path
from typing import List

import tensorflow as tf

from smile import utils
from smile.models.unit.architectures.celeb import paper
from smile.models.unit.input import celeb_input_fn
from smile.models.unit.model import UNIT


def run_training(model_dir: Path,
                 X_train_paths: List[Path],
                 X_test_paths: List[Path],
                 Y_train_paths: List[Path],
                 Y_test_paths: List[Path],
                 **hparams):

    model_dir.mkdir(parents=True, exist_ok=True)

    unit = UNIT(
        celeb_input_fn(X_train_paths, batch_size=hparams["batch_size"]),
        celeb_input_fn(X_test_paths, batch_size=8),
        celeb_input_fn(Y_train_paths, batch_size=hparams["batch_size"]),
        celeb_input_fn(Y_test_paths, batch_size=8),
        paper.private_encoder, paper.shared_encoder,
        paper.shared_decoder, paper.private_decoder,
        paper.discriminator,
        **hparams)

    summary_writer = tf.summary.FileWriter(str(model_dir))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=str(model_dir), save_summaries_secs=30) as sess:
        while not sess.should_stop():
            unit.train_step(sess, summary_writer)


if __name__ == "__main__":
    arg_parser = utils.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
    arg_parser.add_argument("--X-train", nargs="+", required=True, help="Tfrecord train files for first image domain.")
    arg_parser.add_argument("--X-test", nargs="+", required=True, help="Tfrecord test files for first image domain.")
    arg_parser.add_argument("--Y-train", nargs="+", required=True, help="Tfrecord train files for second image domain.")
    arg_parser.add_argument("--Y-test", nargs="+", required=True, help="Tfrecord test files for second image domain.")

    arg_parser.add_hparam("batch-size", default=8, type=int, help="Batch size.")
    arg_parser.add_hparam("lambda_vae_kl", default=0.1, type=float, help="Weight of KL divergence in VAE loss.")
    arg_parser.add_hparam("lambda_vae_nll", default=100.0, type=float, help="Weight of reconstruction in VAE loss.")
    arg_parser.add_hparam("lambda_gan", default=10.0, type=float, help="Weight for GAN losses.")
    arg_parser.add_hparam("lambda_cyclic_kl", default=0.1, type=float, help="Weight of KL divergence in cyclic loss.")
    arg_parser.add_hparam("lambda_cyclic_nll", default=100.0, type=float,
                          help="Weight of reconstruction in cyclic loss.")

    args, hparams = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        model_dir = ROOT_RUNS_DIR / Path(utils.experiment_name("unit", hparams))
    else:
        model_dir = Path(args.model_dir)

    run_training(
        model_dir,
        args.X_train,
        args.X_test,
        args.Y_train,
        args.Y_test,
        **hparams)
