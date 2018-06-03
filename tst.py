import tensorflow as tf

from smile.stargan.data import celeb_input_fn


tf.logging.set_verbosity(tf.logging.INFO)

considered_attributes = ["Smiling", "Big_Nose", "Blond_Hair", "Bangs"]

#img, attributes = celeb_input_fn("/home/john/datasets/celeb/tfrecords/stargan/shard-1", considered_attributes)
ds_iter = celeb_input_fn("/home/john/datasets/celeb/tfrecords/stargan/shard-1", considered_attributes, batch_size=64)
img, attributes = ds_iter.get_next()

# TODO: Want this to be &lt;image&gt;, &lt;indicator vector&gt; considering the considered attributes

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(ds_iter.initializer)

    i, a = sess.run((img, attributes))

    import matplotlib.pyplot as plt

    #for ii, aa in zip(i, a):
    #    print(aa)
    #    plt.imshow(ii)
    #    plt.show()
    #    input()

    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        sess.run((img, attributes))
