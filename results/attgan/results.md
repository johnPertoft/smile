# AttGAN Results
These are results for some AttGAN experiments.

* TODO: Specify the attributes used.
* TODO: Rerun these without augmentation/fix augmentation?

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

