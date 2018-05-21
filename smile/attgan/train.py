from pathlib import Path
from typing import List

import tensorflow as tf

from smile.attgan import AttGAN
from smile.attgan.architectures import celeb
from smile.attgan.input import celeb_input_fn
from smile import utils

tf.logging.set_verbosity(tf.logging.INFO)


def run_training(model_dir: Path,
                 train_tfrecord_paths: List[str],
                 test_tfrecord_paths: List[str],
                 considered_attributes: List[str],
                 **hparams):

    model_dir.mkdir(parents=True, exist_ok=True)

    img_train, attributes_train = \
        celeb_input_fn(train_tfrecord_paths, considered_attributes, batch_size=hparams["batch_size"])

    # TODO: Return sample op for test attributes instead.
    img_test, attributes_test = \
        celeb_input_fn(test_tfrecord_paths, considered_attributes, batch_size=3)

    attgan = AttGAN(
        img_train, attributes_train,
        img_test, attributes_test,
        celeb.paper.encoder,
        celeb.paper.decoder,
        celeb.paper.classifier_discriminator_shared,
        celeb.paper.classifier_private,
        celeb.paper.discriminator_private,
        **hparams)

    summary_writer = tf.summary.FileWriter(str(model_dir))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=str(model_dir), save_summaries_secs=30) as sess:
        pass

if __name__ == "__main__":
    arg_parser = utils.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc")
    arg_parser.add_argument("--train_tfrecords", nargs="+", required=True, help="train tfrecords files for attgan")
    arg_parser.add_argument("--test_tfrecords", nargs="+", required=True, help="test tfrecords files for attgan")

    arg_parser.add_hparam("--batch_size", default=32, type=int, help="Batch size")
    arg_parser.add_hparam("--lambda_rec", default=100.0, type=float, help="Weight of reconstruction loss.")
    arg_parser.add_hparam("--lambda_cls_d", default=1.0, type=float,
                          help="Weight of attribute classification discriminator loss. Relative to GAN loss part.")
    arg_parser.add_hparam("--lambda_cls_g", default=10.0, type=float,
                          help="Weight of attribute classification generator loss. Relative to GAN loss part.")

    args, hparams = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        model_dir = ROOT_RUNS_DIR / Path(utils.experiment_name("attgan", hparams))
    else:
        model_dir = Path(args.model_dir)

    # TODO: Param for this. Handle mutual exclusiveness.
    considered_attributes = ["Smiling", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bald"]

    run_training(
        model_dir,
        args.tfrecords,
        considered_attributes,
        **hparams)
