import argparse
import itertools
from typing import List
from pathlib import Path

import skimage.io
import tensorflow as tf
from tqdm import tqdm

from create_shards import create_shards


def download():
    pass


def create_datasets(dataset_rootdir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths_map = itertools.chain(
            (dataset_rootdir / "trainA").glob("*.jpg"),
            (dataset_rootdir / "valA").glob("*.jpg"),
            (dataset_rootdir / "testA").glob("*.jpg"))

    img_paths_satellite = itertools.chain(
            (dataset_rootdir / "trainB").glob("*.jpg"),
            (dataset_rootdir / "valB").glob("*.jpg"),
            (dataset_rootdir / "testB").glob("*.jpg"))

    create_shards(list(img_paths_map), output_dir / "map", 1000, resize=(256, 256))
    create_shards(list(img_paths_satellite), output_dir / "satellite", 1000, resize=(256, 256))


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset-dir", required=True, help="path to maps dataset")
arg_parser.add_argument("--output-dir", help="optional output directory for tfrecord shard files")
args = arg_parser.parse_args()

create_datasets(Path(args.dataset_dir), Path(args.output_dir or "."))
