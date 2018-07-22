# Attgan Results

TODO: Specify the attributes used.

To see all attgan options:
```bash
$ python -m smile.models.attgan.train --help
```

---

```bash
$ python -m smile.models.attgan.train \
    --train_tfrecords datasets/celeb/tfrecords/all_attributes/train/* \
    --test_tfrecords datasets/celeb/tfrecords/all_attributes/test/* \
    --model_archictecture resnet
```

![attgan](imgs/attgan_resnet.png)