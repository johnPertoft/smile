from pathlib import Path
from typing import Sequence, Union

import skimage.io
import tensorflow as tf
from tqdm import tqdm


_CELEB_A_SHAPE = (218, 178, 3)


def celeb_input_fn(tfrecords_paths, considered_attributes, num_epochs=None, batch_size=64):
    """Return input tensors (img, label)."""

    # Note: One oov bucket for all non considered attributes.
    attribute_index = tf.contrib.lookup.index_table_from_tensor(considered_attributes, num_oov_buckets=1)

    def parse_serialized(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={"img": tf.FixedLenFeature([], tf.string),
                      "attributes": tf.VarLenFeature(tf.string)})

        # Images in [0, 1].
        img = tf.reshape(tf.decode_raw(features["img"], tf.uint8), _CELEB_A_SHAPE)
        img = tf.cast(img, tf.float32)
        img = img / 255

        # Considered attributes as indicator vector.
        attributes = attribute_index.lookup(features["attributes"])
        attributes = tf.sparse_to_indicator(attributes, len(considered_attributes) + 1)  # +1 for the oov bucket.
        attributes = attributes[:-1]  # Skip the oov bucket since those attributes should not be considered.
        attributes = tf.cast(attributes, tf.float32)

        return img, attributes

    def at_least_one_considered_attribute(img, attributes):
        return tf.logical_not(tf.reduce_all(tf.equal(attributes, 0.0)))

    # TODO: Skip the dataset and do it the old way.

    return (tf.data.TFRecordDataset(tfrecords_paths)
            .map(parse_serialized)
            .filter(at_least_one_considered_attribute)
            .shuffle(1024)
            .repeat(num_epochs)
            .batch(batch_size)
            .make_initializable_iterator())
            #.make_one_shot_iterator().get_next())


def prepare_celeb(celeb_root_dir: Union[str, Path]):
    celeb_root_dir = Path(celeb_root_dir)
    attributes_csv = celeb_root_dir / "list_attr_celeba.txt"
    img_dir = celeb_root_dir / "img_align_celeba"
    tfrecords_dir = celeb_root_dir / "tfrecords" / "stargan"
    tfrecords_dir.mkdir(parents=True, exist_ok=True)

    with attributes_csv.open() as attributes:
        next(attributes)  # Skip initial row.
        header_string = next(attributes)
        attribute_columns = header_string.strip().split()

        def get_row(line):
            parts = [l.strip() for l in line.split()]
            path = parts[0]
            active_attributes = [attribute for attribute, p in zip(attribute_columns, parts[1:]) if p == "1"]
            return path, active_attributes

        data_rows = [get_row(line) for line in attributes]

    def write_examples(examples: Sequence[tf.train.Example], shard_path: Path):
        with tf.python_io.TFRecordWriter(str(shard_path)) as record_writer:
            for example in examples:
                record_writer.write(example.SerializeToString())

    examples_per_shard = 1000
    example_buffer = []
    current_shard_index = 0
    for path, active_attributes in tqdm(data_rows):
        img = skimage.io.imread(img_dir / path)

        example = tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            "attributes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[attr.encode("ascii")
                                                                                for attr in active_attributes]))
        }))
        example_buffer.append(example)

        if len(example_buffer) >= examples_per_shard:
            write_examples(example_buffer, tfrecords_dir / f"shard-{current_shard_index}")
            example_buffer = []
            current_shard_index += 1

    if len(example_buffer) > 0:
        write_examples(example_buffer, tfrecords_dir / f"shard-{current_shard_index}")
