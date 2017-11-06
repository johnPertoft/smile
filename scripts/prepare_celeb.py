import argparse
from pathlib import Path
from typing import List

import pandas as pd

from create_shards import create_shards


def create_datasets(attributes_csv_path: Path,
                    img_dir: Path,
                    attribute: str,
                    output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(attributes_csv_path) as attributes_csv:
        next(attributes_csv)  # Skip initial row.

        header_string = next(attributes_csv)
        attribute_columns = header_string.strip().split()
        assert attribute in attribute_columns, f"Invalid attribute. Try any of {attribute_columns}"

        columns = ["path"] + attribute_columns
        df = pd.read_csv(attributes_csv, names=columns, sep=" ", skipinitialspace=True)

    create_shards(
        df[df[attribute] == 1]["path"].apply(lambda p: img_dir / p),
        output_dir / f"{attribute.lower()}",
        1000)

    create_shards(
        df[df[attribute] == -1]["path"].apply(lambda p: img_dir / p),
        output_dir / f"not_{attribute.lower()}",
        1000)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--attributes-csv", required=True, type=Path, help="path to celeb attributes csv")
arg_parser.add_argument("--img-dir", required=True, type=Path, help="path to celeb img directory")
arg_parser.add_argument("--attribute", required=True, help="celeb attribute to split on")
arg_parser.add_argument("--output-dir", type=Path, help="optional output directory for tfrecord shard files")
args = arg_parser.parse_args()

# TODO: Add automatic download of celeb dataset.

create_datasets(
    args.attributes_csv,
    args.img_dir,
    args.attribute,
    args.output_dir or Path("tfrecords"))
