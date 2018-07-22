# Cyclegan Results
These are results for Cyclegan runs with the given settings.

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

![cyclegan](imgs/cyclegan_paper.png)

---

```bash
$ python -m smile.models.cyclegan.train \
    --X-train datasets/celeb/tfrecords/smiling/train/* \
    --X-test datasets/celeb/tfrecords/smiling/test/* \
    --Y-train datasets/celeb/tfrecords/not_smiling/train/* \
    --Y-test datasets/celeb/tfrecords/not_smiling/test/*
```