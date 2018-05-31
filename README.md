# Smile

## Download and Prepare Dataset
For dataset split on a given feature. (Expected by CycleGAN, etc).
```bash
$ python -m smile.utils.data.create_dataset --dataset-dir datasets/celeb --split-attribute Smiling
```

For dataset with attributes included in Tfrecords. (Expected by AttGAN, etc).
```bash
$ python -m smile.utils.data.create_dataset --dataset-dir datasets/celeb --include-attributes
```

## Run training
```bash
$ python -m smile.cyclegan.train \
    --X-train datasets/celeb/tfrecords/smiling/train/* \
    --X-test datasets/celeb/tfrecords/smiling/test/* \
    --Y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --Y-test datasets/celeb/tfrecords/not_smiling/test/*
```

For more options:
```bash
$ python -m smile.cyclegan.train --help
```

## Results
Some cherrypicks.

![alt text](pics/cherrypick1.png)

![alt text](pics/cherrypick2.png)

## TODO

* Consider resizing to 128x128 as most papers do.

### CycleGAN
* WGAN-GP loss
* Densenet architecture(s)
* Spectral normalization
* Progressive growing
* Attention mechanism
* Facial landmarks as supervision

### Other models
* DiscoGAN
* StarGAN (working code, bad results)
* UNIT (working code, bad results)
* MUNIT
* DTN
* XGAN
* DA-gan model
* AttGAN
