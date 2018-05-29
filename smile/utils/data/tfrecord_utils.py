from pathlib import Path
from typing import Iterator, Sequence

import skimage.io
import skimage.transform
import tensorflow as tf
from tqdm import tqdm


def write_examples(examples: Sequence[tf.train.Example], shard_path: Path):
    with tf.python_io.TFRecordWriter(str(shard_path)) as record_writer:
        for example in examples:
            record_writer.write(example.SerializeToString())


def write_shards(shards_dir: Path, examples: Iterator[tf.train.Example]):
    shards_dir.mkdir(parents=True, exist_ok=True)
    examples_per_shard = 1000
    example_buffer = []
    current_shard_index = 0
    for example in tqdm(examples):
        example_buffer.append(example)
        if len(example_buffer) >= examples_per_shard:
            write_examples(example_buffer, shards_dir / f"shard-{current_shard_index:03}")
            example_buffer = []
            current_shard_index += 1
    if len(example_buffer) > 0:
        write_examples(example_buffer, shards_dir / f"shard-{current_shard_index:03}")



# TODO: refactor to remove this one.
def create_shards(img_paths: Sequence[Path], shards_dir: Path, examples_per_shard, resize=None):
    shards_dir.mkdir(parents=True, exist_ok=True)

    def path_to_example(path: Path):
        img = skimage.io.imread(path)
        if resize is not None:
            img = skimage.transform.resize(img, resize)
        return tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
        }))

    def write_examples(examples: Sequence[tf.train.Example], path: Path):
        with tf.python_io.TFRecordWriter(str(path)) as record_writer:
            for example in examples:
                record_writer.write(example.SerializeToString())

    example_buffer = []
    current_shard_index = 0
    for p in tqdm(img_paths):
        example_buffer.append(path_to_example(p))
        if len(example_buffer) >= examples_per_shard:
            write_examples(example_buffer, shards_dir / f"shard-{current_shard_index}")
            example_buffer = []
            current_shard_index += 1
    if len(example_buffer) > 0:
        write_examples(example_buffer, shards_dir / f"shard-{current_shard_index}")
