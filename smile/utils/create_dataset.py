import argparse
from pathlib import Path

import pandas as pd

from .celeb_download_contrib import download_celeb_a
from .tfrecord_utils import create_shards


# TODO: Option to include labels (stargan), currently separate script for this.


def prepare_celeb(root_dir: Path, attribute: str):
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-dir", required=True, type=Path, help="Directory to place dataset in.")
    argparser.add_argument("--attribute", required=True, help="celeb attribute to split on")
    args = argparser.parse_args()

    prepare_celeb(args.dataset_dir, args.attribute)
