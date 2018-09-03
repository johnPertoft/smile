# CycleGAN Results
These are results for some CycleGAN experiments.

TODO: Make gif to show progress.

To see all cyclegan options:
```bash
$ python -m smile.models.cyclegan.train --help
```

---

```bash
$ python -m smile.models.cyclegan.train \
    --X-train datasets/celeb/tfrecords/smiling/train/* \
    --X-test datasets/celeb/tfrecords/smiling/test/* \
    --Y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --Y-test datasets/celeb/tfrecords/not_smiling/test/*
```

![cyclegan](runs/paper-architecture-lambda-cyclic-5.0/testsamples_final.png)

---

```bash
$ python -m smile.models.cyclegan.train \
    --X-train datasets/celeb/tfrecords/smiling/train/* \
    --X-test datasets/celeb/tfrecords/smiling/test/* \
    --Y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --Y-test datasets/celeb/tfrecords/not_smiling/test/*
```

![cyclegan](runs/cyclegan_paper.png)