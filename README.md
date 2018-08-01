# Smile
This repository contains Tensorflow implementations of some models dealing with image translation. Here they are 
applied to the problem of facial attribute editing (e.g. smile to non-smile and vice versa). Some models can only
handle one attribute at a time and some can handle multiple.

## Download and Prepare Dataset
For dataset split on a given feature. (Expected by CycleGAN, etc).
```bash
$ python -m smile.data.prepare.create_dataset --dataset-dir datasets/celeb --split-attribute Smiling
```

For dataset with attributes included in Tfrecords. (Expected by AttGAN, etc).
```bash
$ python -m smile.data.prepare.create_dataset --dataset-dir datasets/celeb --include-attributes
```

## Results

### CycleGAN
```bash
$ python -m smile.models.cyclegan.train \
    --X-train datasets/celeb/tfrecords/smiling/train/* \
    --X-test datasets/celeb/tfrecords/smiling/test/* \
    --Y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --Y-test datasets/celeb/tfrecords/not_smiling/test/*
```

For more options:
```bash
$ python -m smile.models.cyclegan.train --help
```

![cyclegan](results/imgs/cyclegan_paper.png)

Some cherrypicks from full celeb image size.

![alt text](pics/cherrypick1.png)

![alt text](pics/cherrypick2.png)

### AttGAN
TODO: Add command to recreate.

![alt text](pics/attgan.png)

### StarGAN
TODO: Show results.

### UNIT
TODO: Show results.

## TODO
* Better results images per model. Show results at different timesteps.
* Add evaluation method based on classifier of identities? sort of like inception score

### Models / Papers
- [x] CycleGAN
- [x] AttGAN
- [x] UNIT (bad results, needs work)
- [x] StarGAN (bad results, needs work)
- [ ] Sparsely Grouped GAN
- [ ] Fusion GAN
- [ ] DiscoGAN
- [ ] MUNIT
- [ ] XGAN
- [ ] DTN

### General
* Spectral normalization
* Progressive growing
* Attention mechanism, see self-attention GAN
* Facial landmarks as supervision
* Densenet architectures
* https://github.com/yingcong/Facelet_Bank
* Simultaneous vs alternating gradient descent.
* Standardize architecture for comparisons?
