import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--shards-dir", required=True, type=Path, help="Path to directory containing tfrecords shards.")
arg_parser.add_argument("--dataset", required=True, choices=["celeb"])
args = arg_parser.parse_args()


_DATASET_SHAPES = {
    "celeb": (218, 178, 3)
}


def parse_serialized(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"img": tf.FixedLenFeature([], tf.string)})
    img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _DATASET_SHAPES[args.dataset])
    img = tf.cast(img, tf.float32)
    img = img / 255
    return img


img = (tf.data.TFRecordDataset([str(p) for p in args.shards_dir.glob("*")])
    .map(parse_serialized)
    .shuffle(100)
    .make_one_shot_iterator()
    .get_next())

with tf.Session() as sess:
    while True:
        plt.imshow(sess.run(img))
        try:
            plt.waitforbuttonpress()
        except Exception:
            break
