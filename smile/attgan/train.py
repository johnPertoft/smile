from pathlib import Path
from typing import List

import tensorflow as tf

from smile.attgan import AttGAN
from smile.attgan.architectures import celeb
from smile.utils.data.input import input_fn_with_attributes
from smile import utils

tf.logging.set_verbosity(tf.logging.INFO)


def run_training(model_dir: Path,
                 train_tfrecord_paths: List[str],
                 test_tfrecord_paths: List[str],
                 considered_attributes: List[str],
                 **hparams):

    model_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = input_fn_with_attributes(train_tfrecord_paths, considered_attributes, hparams["batch_size"])
    test_dataset = input_fn_with_attributes(test_tfrecord_paths, considered_attributes, 3)

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    img_train, attributes_train = train_iterator.get_next()
    img_test, attributes_test = test_iterator.get_next()

    iterator_initializer = tf.group(train_iterator.initializer, test_iterator.initializer)

    #model_architecture = celeb.paper
    model_architecture = celeb.resnet

    attgan = AttGAN(
        considered_attributes,
        img_train, attributes_train,
        img_test, attributes_test,
        model_architecture.encoder,
        model_architecture.decoder,
        model_architecture.classifier_discriminator_shared,
        model_architecture.classifier_private,
        model_architecture.discriminator_private,
        **hparams)

    summary_writer = tf.summary.FileWriter(str(model_dir))

    scaffold = tf.train.Scaffold(local_init_op=tf.group(
        tf.local_variables_initializer(),
        tf.tables_initializer(),
        iterator_initializer))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    max_training_steps = 150000

    with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            config=config,
            checkpoint_dir=str(model_dir),
            save_summaries_secs=30) as sess:
        while not sess.should_stop():
            i = attgan.train_step(sess, summary_writer, **hparams)
            if i > max_training_steps:
                break

    # Note: tf.train.MonitoredTrainingSession finalizes the graph so can't export from it.
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(str(model_dir)))
        attgan.export(sess, str(model_dir / "export"))


if __name__ == "__main__":
    arg_parser = utils.ArgumentParser()
    arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc")
    arg_parser.add_argument("--train_tfrecords", nargs="+", required=True, help="train tfrecords files for attgan")
    arg_parser.add_argument("--test_tfrecords", nargs="+", required=True, help="test tfrecords files for attgan")

    arg_parser.add_hparam("--batch_size", default=32, type=int, help="Batch size")
    arg_parser.add_hparam("--lambda_rec", default=100.0, type=float, help="Weight of reconstruction loss.")
    arg_parser.add_hparam("--lambda_cls_d", default=1.0, type=float,
                          help="Weight of attribute classification discriminator loss. Relative to GAN loss part.")
    arg_parser.add_hparam("--lambda_cls_g", default=10.0, type=float,
                          help="Weight of attribute classification generator loss. Relative to GAN loss part.")
    arg_parser.add_hparam("--adversarial_loss_type", default="wgan-gp", type=str,
                          help="Adversarial loss function to use.")

    args, hparams = arg_parser.parse_args()

    ROOT_RUNS_DIR = Path("runs")
    if args.model_dir is None:
        model_dir = ROOT_RUNS_DIR / Path(utils.experiment_name("attgan", hparams))
    else:
        model_dir = Path(args.model_dir)

    # TODO: Param for this. Handle mutual exclusiveness?
    considered_attributes = ["Smiling", "Male", "Mustache", "5_o_Clock_Shadow", "Blond_Hair"]

    run_training(
        model_dir,
        args.train_tfrecords,
        args.test_tfrecords,
        considered_attributes,
        **hparams)
