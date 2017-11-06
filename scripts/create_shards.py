from pathlib import Path
from typing import Sequence

import skimage.io
import skimage.transform
import tensorflow as tf
from tqdm import tqdm


def create_shards(img_paths: Sequence[Path], shards_dir: Path, examples_per_shard, resize=None):
    shards_dir.mkdir(exist_ok=True)

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
