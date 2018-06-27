# Things to try to combine

* Decoder part of generator conditioned on attributes.
* Classifier to classify attributes on generated (and real) images to provide training signal.
* PatchGAN discriminator
    * WGAN-gp or LSGAN loss seems best
    * also check DRAGAN
* Progressive Growing of networks
* Attention mechanism
* Facial landmarks as semi supervision
* Combine resnet bottleneck with longer skip connections (unet/densenet) between corresponding scale layers.