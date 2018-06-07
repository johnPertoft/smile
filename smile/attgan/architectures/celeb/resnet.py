import tensorflow


def _resblock(x):
    pass


def encoder(img, is_training, **hparams):
    pass


def decoder(z, attributes, is_training, **hparams):
    pass


def classifier_discriminator_shared(img, is_training, **hparams):
    pass


def classifier_private(h, n_classes, is_training, **hparams):
    # TODO: Stronger classifier should help?
        # regularization
    pass


def discriminator_private(h, is_training, **hparams):
    pass
