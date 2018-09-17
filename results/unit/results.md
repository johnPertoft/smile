# UNIT results
These are results for some UNIT experiments.

To see all UNIT options:
```bash
$ python -m smile.models.unit.train --help
```

---

```bash
$ python -m smile.models.unit.train \
    --x-train datasets/celeb/tfrecords/smiling/train/* \
    --x-test datasets/celeb/tfrecords/smiling/test/* \
    --y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --y-test datasets/celeb/tfrecords/not_smiling/test/* \
    --adversarial_loss lsgan
```

![unit](runs/paper-settings-lsgan/testsamples_final.png)