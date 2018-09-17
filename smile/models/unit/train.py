from pathlib import Path
from typing import List

import tensorflow as tf

import smile.models.unit.architectures
from smile import experiments
from smile.data import dataset
from smile.losses import lsgan_losses
from smile.losses import non_saturating_gan_losses
from smile.losses import wgan_gp_losses
from smile.models.unit import UNIT


tf.logging.set_verbosity(tf.logging.INFO)


arg_parser = experiments.ArgumentParser()
arg_parser.add_argument("--model-dir", required=False, type=Path, help="Directory for checkpoints etc.")
arg_parser.add_argument("--x-train", nargs="+", required=True, help="Tfrecord train files for first image domain.")
arg_parser.add_argument("--x-test", nargs="+", required=True, help="Tfrecord test files for first image domain.")
arg_parser.add_argument("--y-train", nargs="+", required=True, help="Tfrecord train files for second image domain.")
arg_parser.add_argument("--y-test", nargs="+", required=True, help="Tfrecord test files for second image domain.")
arg_parser.add_argument("--steps", default=200000, type=int, help="Number of train steps.")

arg_parser.add_hparam("--batch_size", default=16, type=int, help="Batch size.")
arg_parser.add_hparam("--model_architecture", default="paper", help="Model architecture.")
arg_parser.add_hparam("--adversarial_loss", default="nsgan", type=str, help="Adversarial loss function to use.")
arg_parser.add_hparam("lambda_vae_kl", default=0.1, type=float, help="Weight of KL divergence in VAE loss.")
arg_parser.add_hparam("lambda_vae_rec", default=100.0, type=float, help="Weight of reconstruction in VAE loss.")
arg_parser.add_hparam("lambda_adv", default=10.0, type=float, help="Weight for adversarial losses.")
arg_parser.add_hparam("lambda_cyclic_kl", default=0.1, type=float, help="Weight of KL divergence in cyclic loss.")
arg_parser.add_hparam("lambda_cyclic_rec", default=100.0, type=float, help="Weight of reconstruction in cyclic loss.")

args, hparams = arg_parser.parse_args()


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
    model_architecture = smile.models.unit.architectures.paper
else:
    raise ValueError("Invalid model architecture.")


if hparams["adversarial_loss"] == "lsgan":
    adversarial_loss_fn = lsgan_losses
    hparams["n_discriminator_iters"] = 1
elif hparams["adversarial_loss"] == "nsgan":
    adversarial_loss_fn = non_saturating_gan_losses
    hparams["n_discriminator_iters"] = 1
elif hparams["adversarial_loss"] == "wgan-gp":
    adversarial_loss_fn = wgan_gp_losses
    hparams["n_discriminator_iters"] = 5
    hparams["wgan_gp_lambda"] = 10.0
else:
    raise ValueError("Invalid adversarial loss fn.")


unit = UNIT(
    a_train=input_fn(args.x_train, hparams["batch_size"]),
    a_test=input_fn(args.x_test, 3),
    a_test_static=first_n(args.x_test, 10),
    b_train=input_fn(args.y_train, hparams["batch_size"]),
    b_test=input_fn(args.y_test, 3),
    b_test_static=first_n(args.y_test, 10),
    private_encoder_fn=model_architecture.encoder_private,
    shared_encoder_fn=model_architecture.encoder_shared,
    shared_decoder_fn=model_architecture.decoder_shared,
    private_decoder_fn=model_architecture.decoder_private,
    discriminator_fn=model_architecture.discriminator,
    adversarial_loss_fn=adversarial_loss_fn,
    **hparams)


experiments.run_experiment(
    model_dir=args.model_dir or experiments.ROOT_RUNS_DIR / experiments.experiment_name("unit", hparams),
    model=unit,
    n_training_step=args.steps)
