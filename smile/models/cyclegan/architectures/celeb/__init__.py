from smile.models.cyclegan.architectures.celeb.full.densenet import densenet_generator, densenet_generator2
from smile.models.cyclegan.architectures.celeb.full.paper import paper_discriminator, paper_generator
from smile.models.cyclegan.architectures.celeb.full.unet import unet_generator
from smile.models.cyclegan.architectures.celeb import paper

"""
GENERATORS = {
    "paper": paper_generator,
    "unet": unet_generator,
    "densenet": densenet_generator,
    "densenet2": densenet_generator2
}

DISCRIMINATORS = {
    "paper": paper_discriminator
}
"""

GENERATORS = {
    "paper": paper.generator,
}

DISCRIMINATORS = {
    "paper": paper.discriminator
}
