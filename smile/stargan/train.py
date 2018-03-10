import argparse
import datetime
from pathlib import Path

import tensorflow as tf

from smile.stargan import StarGAN


tf.logging.set_verbosity(tf.logging.INFO)


def run_training(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)

    star_gan = StarGAN()

    summary_writer = tf.summary.FileWriter(str(model_dir))

    with tf.train.MonitoredTrainingSession(checkpoint_dir=str(model_dir), save_summaries_secs=30) as sess:
        while not sess.should_stop():
            star_gan.train_step(sess, summary_writer)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="directory for checkpoints etc")
    #arg_parser.add_argument("-X", nargs="+", required=True, help="tfrecord files for first image domain")
    #arg_parser.add_argument("-Y", nargs="+", required=True, help="tfrecord files for second image domain")
    #arg_parser.add_argument("--batch-size", required=True, type=int, help="batch size")
    args = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        model_dir = ROOT_RUNS_DIR / Path(f"stargan_{timestamp}")
    else:
        model_dir = Path(args.model_dir)

    run_training(model_dir)
