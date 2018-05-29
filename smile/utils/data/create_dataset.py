import argparse
from pathlib import Path

import pandas as pd
import skimage.io
import tensorflow as tf

from smile.utils.data.celeb_download_contrib import download_celeb_a
from smile.utils.data.tfrecord_utils import create_shards, write_shards


def prepare_celeb(root_dir: Path, attribute: str):
    """
    Downloads the celeba dataset and creates tfrecords that are split according to
    the given attribute and into a train- and test-set.
    :param root_dir: Directory where dataset will be placed.
    :param attribute: Attribute to split on.
    :return: None
    """

    # TODO: Check if things exist already.
    root_dir.mkdir(parents=True, exist_ok=True)
    download_celeb_a(str(root_dir / "raw"))

    with open(root_dir / "raw" / "list_attr_celeba.txt") as attributes_csv:
        next(attributes_csv)  # Skip initial row.

        header_string = next(attributes_csv)
        attribute_columns = header_string.strip().split()
        assert attribute in attribute_columns, f"Invalid attribute. Try any of {attribute_columns}"

        columns = ["path"] + attribute_columns
        df = pd.read_csv(attributes_csv, names=columns, sep=" ", skipinitialspace=True)

    def determine_partition(p):
        # Note: These values are from list_eval_celeb.txt
        img, _ = p.split(".")
        n = int(img)
        if n < 162771:
            return "train"
        elif n < 182638:
            return "train"  # Adding validation set to train set for our purposes.
        else:
            return "test"

    df["partition"] = df["path"].apply(determine_partition)

    img_dir = root_dir / "raw" / "img_align_celebA"
    tfrecords_root_dir = root_dir / "tfrecords"

    def pick_paths(active, partition):
        paths = df[(df[attribute] == active) & (df["partition"] == partition)]["path"]
        return paths.apply(lambda p: img_dir / p)

    create_shards(
        pick_paths(1, "train"),
        tfrecords_root_dir / f"{attribute.lower()}" / "train",
        1000)

    create_shards(
        pick_paths(1, "test"),
        tfrecords_root_dir / f"{attribute.lower()}" / "test",
        1000)

    create_shards(
        pick_paths(-1, "train"),
        tfrecords_root_dir / f"not_{attribute.lower()}" / "train",
        1000)

    create_shards(
        pick_paths(-1, "test"),
        tfrecords_root_dir / f"not_{attribute.lower()}" / "test",
        1000)


def prepare_celeb_with_attributes(root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Check if things already exist.

    raw_celeb_dir = root_dir / "raw"
    if not raw_celeb_dir.exists():
        download_celeb_a(str(raw_celeb_dir))

    with (raw_celeb_dir / "list_attr_celeba.txt").open() as attributes_csv:
        next(attributes_csv)  # Initial row is not needed.
        header_string = next(attributes_csv)
        attribute_columns = header_string.strip().split()

        def get_row(line):
            parts = [l.strip() for l in line.split()]
            path = parts[0]
            active_attributes = [attribute for attribute, p in zip(attribute_columns, parts[1:]) if p == "1"]
            return path, active_attributes

        data_rows = [get_row(line) for line in attributes_csv]

    tfrecords_dir = root_dir / "tfrecords" / "all_attributes"
    tfrecords_dir.mkdir(parents=True, exist_ok=True)

    # Note: This value is from list_eval_celeb.txt and here
    # we include both train and eval set as the train set.
    split_index = 182638
    train_data_rows = data_rows[:split_index]
    test_data_rows = data_rows[split_index:]

    def tf_example(data_row):
        path, active_attributes = data_row
        img = skimage.io.imread(raw_celeb_dir / "img_align_celebA" / path)
        example = tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            "attributes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[attr.encode("ascii")
                                                                                for attr in active_attributes]))
        }))
        return example

    train_examples = (tf_example(dr) for dr in train_data_rows)
    test_examples = (tf_example(dr) for dr in test_data_rows)

    write_shards(tfrecords_dir / "train", train_examples)
    write_shards(tfrecords_dir / "test", test_examples)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-dir", required=True, type=Path, help="Directory to place dataset in.")
    argparser.add_argument("--attribute", required=True, help="celeb attribute to split on")
    args = argparser.parse_args()

    if args.attribute.lower() == "all":
        prepare_celeb_with_attributes(args.dataset_dir)
    else:
        prepare_celeb(args.dataset_dir, args.attribute)
