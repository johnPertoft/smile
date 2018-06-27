import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

from smile.models.stargan import StarGAN
from smile.models.stargan.architectures import celeb
from smile.models.stargan.data import celeb_input_iterator


tf.logging.set_verbosity(tf.logging.INFO)


def run_celeb_training(model_dir: Path,
                       tfrecord_paths: List[str],
                       considered_attributes: List[str],
                       hparams: Dict[str, Any]):

    model_dir.mkdir(parents=True, exist_ok=True)

    celeb_iterator = celeb_input_iterator(
        tfrecord_paths,
        considered_attributes,
        batch_size=hparams["batch_size"])

    init = tf.group(tf.tables_initializer(), celeb_iterator.initializer)
    imgs, attributes = celeb_iterator.get_next()

    star_gan = StarGAN(
        imgs,
        attributes,
        celeb.generator,
        celeb.discriminator,
        hparams["lambda_cls"],
        hparams["lambda_rec"])

    summary_writer = tf.summary.FileWriter(str(model_dir))

    #with tf.train.MonitoredTrainingSession(checkpoint_dir=str(model_dir), save_summaries_secs=30) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(init)

        #while not sess.should_stop():
        while True:
            star_gan.train_step(sess, summary_writer)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="directory for checkpoints etc")
    arg_parser.add_argument("--tfrecords", nargs="+", required=True, help="tfrecords files for stargan")
    args = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        model_dir = ROOT_RUNS_DIR / Path(f"stargan_{timestamp}")
    else:
        model_dir = Path(args.model_dir)

    # TODO: Get this from program args.
    hparams = {
        "batch_size": 16,
        "lambda_cls": 1.0,
        "lambda_rec": 10.0
    }

    considered_attributes = ["Smiling", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bald"]

    run_celeb_training(
        model_dir,
        args.tfrecords,
        considered_attributes,
        hparams)
