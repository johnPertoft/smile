from pathlib import Path
from typing import List

import tensorflow as tf

from smile import utils
from smile.data.celeb import img_dataset
from smile.models.cyclegan import CycleGAN
from smile.models.cyclegan.architectures import celeb


tf.logging.set_verbosity(tf.logging.INFO)


def run_training(model_dir: Path,
                 X_train_paths: List[Path],
                 X_test_paths: List[Path],
                 Y_train_paths: List[Path],
                 Y_test_paths: List[Path],
                 **hparams):

    model_dir.mkdir(parents=True, exist_ok=True)

    def input_fn(paths, batch_size):
        ds = img_dataset(paths, batch_size, crop_and_rescale=True)
        img = ds.make_one_shot_iterator().get_next()
        return img

    cycle_gan = CycleGAN(
        input_fn(X_train_paths, batch_size=hparams["batch_size"]),
        input_fn(X_test_paths, batch_size=4),
        input_fn(Y_train_paths, batch_size=hparams["batch_size"]),
        input_fn(Y_test_paths, batch_size=4),
        celeb.GENERATORS[hparams["generator_architecture"]],
        celeb.DISCRIMINATORS[hparams["discriminator_architecture"]],
        **hparams)

    summary_writer = tf.summary.FileWriter(str(model_dir))

    max_training_steps = 200000

    with tf.train.MonitoredTrainingSession(checkpoint_dir=str(model_dir), save_summaries_secs=30) as sess:
        while not sess.should_stop():
            i = cycle_gan.train_step(sess, summary_writer)
            if i > max_training_steps:
                break

        cycle_gan.generate_samples(sess, str(model_dir / "testsamples.png"))

    # Note: tf.train.MonitoredTrainingSession finalizes the graph so can't export from it.
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(str(model_dir)))
        cycle_gan.export(sess, str(model_dir / "export"))


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
        model_dir = ROOT_RUNS_DIR / Path(utils.experiment_name("cyclegan", hparams))
    else:
        model_dir = Path(args.model_dir)

    run_training(
        model_dir,
        args.X_train,
        args.X_test,
        args.Y_train,
        args.Y_test,
        **hparams)

    # TODO: also see https://arxiv.org/pdf/1611.05507.pdf
    # TODO: Preprocess data to extract only face?
    # TODO: Add attention?
    # TODO: Try wgan-gp loss?
    # TODO: Try different upsampling techniques
    # TODO: Try other architectures.
    # TODO: Pass in facial landmark information during training
    # TODO: Allow gpu mem growth
