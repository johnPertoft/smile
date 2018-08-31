from smile.models.cyclegan.architectures import paper


GENERATORS = {
    "paper": paper.generator,
}

DISCRIMINATORS = {
    "paper": paper.discriminator
}
