import tensorflow as tf

from smile.data import dataset
from smile.models.cyclegan import architectures
from smile.models.cyclegan import CycleGAN
from smile import experiments


tf.logging.set_verbosity(tf.logging.INFO)


arg_parser = experiments.ArgumentParser()
arg_parser.add_argument("--model-dir", required=False, help="Directory for checkpoints etc.")
arg_parser.add_argument("--x-train", nargs="+", required=True, help="Tfrecord train files for first image domain.")
arg_parser.add_argument("--x-test", nargs="+", required=True, help="Tfrecord test files for first image domain.")
arg_parser.add_argument("--y-train", nargs="+", required=True, help="Tfrecord train files for second image domain.")
arg_parser.add_argument("--y-test", nargs="+", required=True, help="Tfrecord test files for second image domain.")
arg_parser.add_argument("--steps", default=200000, type=int, help="Number of train steps.")

# Required hparams.
arg_parser.add_hparam("batch-size", default=16, type=int, help="Batch size.")
arg_parser.add_hparam("generator-architecture", default="paper", help="Architecture for generator network.")
arg_parser.add_hparam("discriminator-architecture", default="paper", help="Architecture for discriminator network.")
arg_parser.add_hparam("lambda-cyclic", default=5.0, type=float, help="Cyclic consistency loss weight.")
arg_parser.add_hparam("use-history", action="store_true",
                      help="Whether a history of generated images should be shown to the discriminator.")

# Conditional hparams.
arg_parser.add_hparam("growth-rate", default=16, type=int, help="Growth rate for densenet architecture.")

args, hparams = arg_parser.parse_args()


# TODO: Allow for restart of training. Load hparams from file.
# TODO: Make CycleGAN implement the Model interface.
# TODO: Specify/use data augmentation?
# TODO: Use hparam for architecture.
# TODO: Add readme for model?
# TODO: mutex group for conditional arguments.
# TODO: Put some of these input fns in shared component. Take dataset_fn as argument.


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


cyclegan = CycleGAN(
    a_train=input_fn(args.x_train, hparams["batch_size"]),
    a_test=input_fn(args.x_test, 3),
    a_test_static=first_n(args.x_test, 10),
    b_train=input_fn(args.y_train, hparams["batch_size"]),
    b_test=input_fn(args.y_test, 3),
    b_test_static=first_n(args.y_test, 10),
    generator_fn=architectures.GENERATORS[hparams["generator_architecture"]],
    discriminator_fn=architectures.DISCRIMINATORS[hparams["discriminator_architecture"]],
    **hparams)


experiments.run_experiment(
    model_dir=experiments.ROOT_RUNS_DIR / experiments.experiment_name("cyclegan", hparams),
    model=cyclegan,
    n_training_step=args.steps)
