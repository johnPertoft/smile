import tensorflow as tf

from smile import experiments
from smile.data import dataset
from smile.losses import lsgan_losses
from smile.losses import non_saturating_gan_losses
from smile.losses import wgan_gp_losses
from smile.models.discogan import architectures
from smile.models.discogan import DiscoGAN


tf.logging.set_verbosity(tf.logging.INFO)


arg_parser = experiments.ArgumentParser()
arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
arg_parser.add_argument("--x-train", nargs="+", required=True, help="Tfrecord train files for first image domain.")
arg_parser.add_argument("--x-test", nargs="+", required=True, help="Tfrecord test files for first image domain.")
arg_parser.add_argument("--y-train", nargs="+", required=True, help="Tfrecord train files for second image domain.")
arg_parser.add_argument("--y-test", nargs="+", required=True, help="Tfrecord test files for second image domain.")
arg_parser.add_argument("--steps", default=200000, type=int, help="Number of train steps.")

arg_parser.add_hparam("--batch_size", default=64, type=int, help="Batch size.")
arg_parser.add_hparam("--model_architecture", default="paper", help="Model architecture.")
arg_parser.add_hparam("--adversarial_loss", default="lsgan", type=str, help="Adversarial loss function to use.")

args, hparams = arg_parser.parse_args()


# TODO: default gan loss should be vanilla?
# TODO: default reconstruction loss?
# TODO: minibatch size of 200?
# TODO: Weighting of different loss parts?


def input_fn(paths, batch_size):
    ds = dataset(paths, crop_and_rescale=True)
    ds = ds.shuffle(1024)
    ds = ds.repeat(None)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    img = ds.make_one_shot_iterator().get_next()
    return img


def first_n(paths, n):
    ds = dataset(paths, crop_and_rescale=True)
    ds = ds.batch(n)
    ds = ds.take(1)
    ds = ds.repeat(None)
    img = ds.make_one_shot_iterator().get_next()
    return img


if hparams["model_architecture"] == "paper":
    model_architecture = architectures.paper
else:
    raise ValueError("Invalid model architecture.")

if hparams["adversarial_loss"] == "nsgan":
    adversarial_loss_fn = non_saturating_gan_losses
    hparams["n_discriminator_iters"] = 1
elif hparams["adversarial_loss"] == "lsgan":
    adversarial_loss_fn = lsgan_losses
    hparams["n_discriminator_iters"] = 1
elif hparams["adversarial_loss"] == "wgan-gp":
    adversarial_loss_fn = wgan_gp_losses
    hparams["n_discriminator_iters"] = 5
    hparams["wgan_gp_lambda"] = 10.0
else:
    raise ValueError("Invalid adversarial loss fn.")


discogan = DiscoGAN(
    a_train=input_fn(args.x_train, hparams["batch_size"]),
    a_test=input_fn(args.x_test, 3),
    a_test_static=first_n(args.x_test, 10),
    b_train=input_fn(args.y_train, hparams["batch_size"]),
    b_test=input_fn(args.y_test, 3),
    b_test_static=first_n(args.y_test, 10),
    generator_fn=model_architecture.generator,
    discriminator_fn=model_architecture.discriminator,
    adversarial_loss_fn=adversarial_loss_fn,
    **hparams)


experiments.run_experiment(
    model_dir=args.model_dir or experiments.ROOT_RUNS_DIR / experiments.experiment_name("discogan", hparams),
    model=discogan,
    n_training_step=args.steps)
