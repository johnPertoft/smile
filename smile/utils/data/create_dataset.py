import argparse
import contextlib
from pathlib import Path
from typing import Any, Iterator, List, Sequence

import skimage.io
import tensorflow as tf
from tqdm import tqdm

from .celeb_a_hq_deltas_download import download_celeb_a_hq_delta_files
from .contrib.celeb_a_download import download_celeb_a


def _maybe_download(root_dir: Path, hq: bool=False) -> Path:
    root_dir.mkdir(parents=True, exist_ok=True)
    raw_celeb_dir = root_dir / "raw"

    if not raw_celeb_dir.exists():
        tf.logging.info("Downloading celeb-a dataset.")
        download_celeb_a(str(raw_celeb_dir))

    if hq:
        _maybe_download_celeb_hq_delta_files(raw_celeb_dir / "celeb_hq_delta_files")

    return raw_celeb_dir


def _maybe_download_celeb_hq_delta_files(celeb_hq_delta_files_dir: Path) -> Path:
    if not celeb_hq_delta_files_dir.exists():
        tf.logging.info("Downloading celeb-a hq delta files.")
        download_celeb_a_hq_delta_files(celeb_hq_delta_files_dir)
        exit()


@contextlib.contextmanager
def _attributes_csv_iterator(root_dir: Path):
    with (root_dir / "raw" / "list_attr_celeba.txt").open() as attributes_csv:
        next(attributes_csv)  # Initial row is not needed.
        header_string = next(attributes_csv)
        attribute_columns = header_string.strip().split()

        yield attribute_columns, attributes_csv


def _train_test_split(data_rows: List[Any]):
    # Note: This value is from list_eval_celeb.txt and here
    # we include both train and eval set as the train set.
    split_index = 182638
    return data_rows[:split_index], data_rows[split_index:]


def _write_examples(examples: Sequence[tf.train.Example], shard_path: Path):
    with tf.python_io.TFRecordWriter(str(shard_path)) as record_writer:
        for example in examples:
            record_writer.write(example.SerializeToString())


def _write_shards(shards_dir: Path, examples: Iterator[tf.train.Example], n_examples):
    shards_dir.mkdir(parents=True, exist_ok=True)
    examples_per_shard = 1000
    example_buffer = []
    current_shard_index = 0
    with tqdm(total=n_examples) as pbar:
        for example in examples:
            example_buffer.append(example)
            if len(example_buffer) >= examples_per_shard:
                _write_examples(example_buffer, shards_dir / f"shard-{current_shard_index:03}")
                example_buffer = []
                current_shard_index += 1
            pbar.update()
        if len(example_buffer) > 0:
            _write_examples(example_buffer, shards_dir / f"shard-{current_shard_index:03}")


def prepare_celeb(root_dir: Path, attribute: str):
    """
    Downloads the celeba dataset and creates tfrecords that are split according to
    the given attribute and into a train- and test-set.
    :param root_dir: Directory where dataset will be downloaded to and tfrecords will be created under.
    :param attribute: Attribute to split on.
    :return: None
    """

    raw_celeb_dir = _maybe_download(root_dir)
    tfrecords_dir = root_dir / "tfrecords"
    with_attribute_dir = tfrecords_dir / f"{attribute.lower()}"
    without_attribute_dir = tfrecords_dir / f"not_{attribute.lower()}"
    if with_attribute_dir.exists() and without_attribute_dir.exists():
        tf.logging.info("Tfrecords already exist.")
        return

    with _attributes_csv_iterator(root_dir) as (attribute_columns, attributes_csv):
        assert attribute in attribute_columns, f"Invalid attribute. Try any of {attribute_columns}"
        attribute_idx = attribute_columns.index(attribute)

        def get_row(line):
            parts = [l.strip() for l in line.split()]
            path = parts[0]
            has_attribute = parts[1 + attribute_idx] == "1"
            return path, has_attribute

        data_rows = [get_row(line) for line in attributes_csv]

    train_data_rows, test_data_rows = _train_test_split(data_rows)
    train_paths_with_attribute = [p for p, has_attr in train_data_rows if has_attr]
    train_paths_without_attribute = [p for p, has_attr in train_data_rows if not has_attr]
    test_paths_with_attribute = [p for p, has_attr in test_data_rows if has_attr]
    test_paths_without_attribute = [p for p, has_attr in test_data_rows if not has_attr]

    def tf_example(path):
        img = skimage.io.imread(raw_celeb_dir / "img_align_celebA" / path)
        example = tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
        }))
        return example

    def write_shards(shards_dir, paths):
        _write_shards(shards_dir, (tf_example(p) for p in paths), len(paths))

    write_shards(with_attribute_dir / "train", train_paths_with_attribute)
    write_shards(with_attribute_dir / "test", test_paths_with_attribute)
    write_shards(without_attribute_dir / "train", train_paths_without_attribute)
    write_shards(without_attribute_dir / "test", test_paths_without_attribute)


def prepare_celeb_with_attributes(root_dir: Path):
    """
    Downloads the celeba dataset and creates tfrecords that are split according to
    the given attribute and into a train- and test-set.
    :param root_dir: Directory where dataset will be downloaded to and tfrecords will be created under.
    :param attribute: Attribute to split on.
    :return: None
    """

    raw_celeb_dir = _maybe_download(root_dir)
    tfrecords_dir = root_dir / "tfrecords" / "all_attributes"
    if tfrecords_dir.exists():
        tf.logging.info("Tfrecords already exist.")
        return

    with _attributes_csv_iterator(root_dir) as (attribute_columns, attributes_csv):
        def get_row(line):
            parts = [l.strip() for l in line.split()]
            path = parts[0]
            active_attributes = [attribute for attribute, p in zip(attribute_columns, parts[1:]) if p == "1"]
            return path, active_attributes

        data_rows = [get_row(line) for line in attributes_csv]

    train_data_rows, test_data_rows = _train_test_split(data_rows)

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

    tfrecords_dir.mkdir(parents=True, exist_ok=True)
    _write_shards(tfrecords_dir / "train", train_examples, len(train_data_rows))
    _write_shards(tfrecords_dir / "test", test_examples, len(test_data_rows))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-dir", required=True, type=Path, help="Directory to place dataset in.")
    mutex_group = argparser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument("--include-attributes", action="store_true",
                             help="Whether to include attributes in tfrecords.")
    mutex_group.add_argument("--split-attribute", help="Celeb-a attribute to split on.")
    argparser.add_argument("--hq", action="store_true", help="Whether to create the HQ variant of the dataset.")
    args = argparser.parse_args()

    # TODO: Use the hq dataset variant as well.

    if args.include_attributes:
        prepare_celeb_with_attributes(args.dataset_dir)
    else:
        prepare_celeb(args.dataset_dir, args.split_attribute)
