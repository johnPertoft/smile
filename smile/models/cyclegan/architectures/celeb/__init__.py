from .densenet import densenet_generator, densenet_generator2
from .paper import paper_discriminator, paper_generator
from .unet import unet_generator


GENERATORS = {
    "paper": paper_generator,
    "unet": unet_generator,
    "densenet": densenet_generator,
    "densenet2": densenet_generator2
}

DISCRIMINATORS = {
    "paper": paper_discriminator
}