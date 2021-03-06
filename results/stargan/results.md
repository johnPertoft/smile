# StarGAN Results
These are results for some StarGAN experiments.

To see all StarGAN options:
```bash
$ python -m smile.models.stargan.train --help
```

```bash
$ python -m smile.models.stargan.train \
    --train-tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test-tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --considered-attributes Smiling Male Mustache Blond_Hair
```

![stargan](runs/paper-architecture-wgan-gp-lambda-rec-10.0-lambda-cls-1.0/testsamples_final.png)

---

```bash
$ python -m smile.models.stargan.train \
    --train-tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test-tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --considered-attributes Smiling Male Mustache Blond_Hair \
    --adversarial_loss lsgan
```

![stargan](runs/paper-architecture-lsgan-lambda-rec-10.0-lambda-cls.1.0/testsamples_final.png)

---

```bash
$ python -m smile.models.stargan.train \
    --train-tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test-tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --considered-attributes Smiling Male Mustache Blond_Hair \
    --lambda-rec 5.0
```

![stargan](runs/paper-architecture-wgan-gp-lambda-rec-5.0-lambda-cls-1.0/testsamples_final.png)
