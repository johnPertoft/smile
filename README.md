# Smile

## Prerequisites
* Download [celeb dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Prepare tfrecords
```bash
$ python scripts/prepare_celeb.py --attributes-csv path/to/list_attr_celeba.txt --img-dir path/to/img_align_celeba --output-dir path/to/output --attribute Smiling
```

## Run training
