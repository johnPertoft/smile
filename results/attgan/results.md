# AttGAN Results
These are results for some AttGAN experiments.

To see all attgan options:
```bash
$ python -m smile.models.attgan.train --help
```

---

```bash
$ python -m smile.models.attgan.train \
    --train-tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test-tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --considered-attributes Smiling Male Mustache Blond_Hair
```

![attgan](runs/paper-architecture-wgan-gp-lambda-rec-100-1-10/testsamples_final.png)

---

```bash
$ python -m smile.models.attgan.train \
    --train-tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test-tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --considered-attributes Smiling Male Mustache Blond_Hair \
    --lambda_rec 50.0 \
    --lambda_cls_g 25.0
```

![attgan](runs/paper-architecture-wgan-gp-lambda-rec-50-1-25/testsamples_final.png)

