import tensorflow as tf

import smile.models.stargan.architectures
from smile.data import dataset_with_attributes
from smile.losses import lsgan_losses
from smile.losses import wgan_gp_losses
from smile.models.stargan import StarGAN
from smile import experiments


tf.logging.set_verbosity(tf.logging.INFO)


arg_parser = experiments.ArgumentParser()
arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
arg_parser.add_argument("--train-tfrecords", nargs="+", required=True, help="Tfrecord train files.")
arg_parser.add_argument("--test-tfrecords", nargs="+", required=True, help="Tfrecord test files.")
arg_parser.add_argument("--considered-attributes", nargs="+", required=True, help="Celeb-a attributes to consider.")
arg_parser.add_argument("--steps", default=200000, type=int, help="Number of train steps.")

arg_parser.add_hparam("--batch_size", default=32, type=int, help="Batch size")
arg_parser.add_hparam("--lambda_rec", default=10.0, type=float, help="Weight of reconstruction loss.")
arg_parser.add_hparam("--lambda_cls", default=1.0, type=float, help="Weight of classification loss.")

args, hparams = arg_parser.parse_args()


def create_dataset(paths):
    return dataset_with_attributes(
        paths,
        args.considered_attributes,
        crop_and_rescale=True,
        filter_examples_without_attributes=True)


train_iterator = (create_dataset(args.train_tfrecords)
    .shuffle(1024)
    .repeat(None)
    .batch(hparams["batch_size"])
    .prefetch(2)
    .make_initializable_iterator())

_test_ds = create_dataset(args.test_tfrecords)

test_iterator = (_test_ds
    .shuffle(1024)
    .repeat(None)
    .batch(3)
    .prefetch(2)
    .make_initializable_iterator())

test_static_iterator = (_test_ds.batch(6)
    .take(1)
    .repeat(None)
    .make_initializable_iterator())

init_op = tf.group(
    train_iterator.initializer,
    test_iterator.initializer,
    test_static_iterator.initializer)

img_train, attributes_train = train_iterator.get_next()
img_test, attributes_test = test_iterator.get_next()
img_test_static, attributes_test_static = test_static_iterator.get_next()


if hparams["model_architecture"] == "paper":
    model_architecture = smile.models.stargan.architectures.paper
else:
    raise ValueError("Invalid model architecture.")


if hparams["adversarial_loss_fn"] == "lsgan":
    adversarial_loss_fn = lsgan_losses
elif hparams["adversarial_loss_fn"] == "wgan-gp":
    adversarial_loss_fn = wgan_gp_losses
    hparams["n_discriminator_iters"] = 5
    hparams["wgan_gp_lambda"] = 10.0
else:
    raise ValueError("Invalid adversarial loss fn.")


stargan = StarGAN(
    attribute_names=args.considered_attributes,
    img=img_train,
    attributes=attributes_train,
    img_test=img_test,
    attributes_test=attributes_test,
    img_test_static=img_test_static,
    attributes_test_static=attributes_test_static,
    generator_fn=model_architecture.generator,
    classifier_discriminator_shared_fn=None,
    classifier_private_fn=None,
    discriminator_private_fn=None,
    adversarial_loss_fn=adversarial_loss_fn,
    **hparams)


experiments.run_experiment(
    model_dir=experiments.ROOT_RUNS_DIR / experiments.experiment_name("stargan", hparams),
    model=stargan,
    n_training_step=args.steps,
    custom_init_op=init_op)
