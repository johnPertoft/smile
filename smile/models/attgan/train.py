import tensorflow as tf

import smile.models.attgan.architectures
from smile.data import dataset_with_attributes
from smile.losses import lsgan_losses
from smile.losses import wgan_gp_losses
from smile.models.attgan import AttGAN
from smile import experiments


tf.logging.set_verbosity(tf.logging.INFO)


arg_parser = experiments.ArgumentParser()
arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
arg_parser.add_argument("--train-tfrecords", nargs="+", required=True, help="Tfrecord train files.")
arg_parser.add_argument("--test-tfrecords", nargs="+", required=True, help="Tfrecord test files.")
arg_parser.add_argument("--considered-attributes", nargs="+", required=True, help="Celeb-a attributes to consider.")
arg_parser.add_argument("--steps", default=200000, type=int, help="Number of train steps.")

arg_parser.add_hparam("--batch_size", default=32, type=int, help="Batch size")
arg_parser.add_hparam("--lambda_rec", default=100.0, type=float, help="Weight of reconstruction loss.")
arg_parser.add_hparam("--lambda_cls_d", default=1.0, type=float,
                      help="Weight of attribute classification discriminator loss. Relative to GAN loss part.")
arg_parser.add_hparam("--lambda_cls_g", default=10.0, type=float,
                      help="Weight of attribute classification generator loss. Relative to GAN loss part.")
arg_parser.add_hparam("--adversarial_loss", default="wgan-gp", type=str,
                      help="Adversarial loss function to use.")
arg_parser.add_hparam("--model_architecture", default="paper", help="Model architecture.")

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
    model_architecture = smile.models.attgan.architectures.paper
elif hparams["model_architecture"] == "resnet":
    model_architecture = smile.models.attgan.architectures.resnet
else:
    raise ValueError("Invalid model architecture.")


if hparams["adversarial_loss"] == "lsgan":
    adversarial_loss_fn = lsgan_losses
elif hparams["adversarial_loss"] == "wgan-gp":
    adversarial_loss_fn = wgan_gp_losses
    hparams["n_discriminator_iters"] = 5
    hparams["wgan_gp_lambda"] = 10.0
else:
    raise ValueError("Invalid adversarial loss fn.")

attgan = AttGAN(
    attribute_names=args.considered_attributes,
    img=img_train,
    attributes=attributes_train,
    img_test=img_test,
    attributes_test=attributes_test,
    img_test_static=img_test_static,
    attributes_test_static=attributes_test_static,
    encoder_fn=model_architecture.encoder,
    decoder_fn=model_architecture.decoder,
    classifier_discriminator_shared_fn=model_architecture.classifier_discriminator_shared,
    classifier_private_fn=model_architecture.classifier_private,
    discriminator_private_fn=model_architecture.discriminator_private,
    adversarial_loss_fn=adversarial_loss_fn,
    **hparams)


experiments.run_experiment(
    model_dir=experiments.ROOT_RUNS_DIR / experiments.experiment_name("attgan", hparams),
    model=attgan,
    n_training_step=args.steps,
    custom_init_op=init_op)
