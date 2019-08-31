import tensorflow as tf
import constants
import os
import sys
import inference
import class_parser as cp
import numpy as np
import platform
import matplotlib.pyplot as plt

print('system='+platform.system())
print('DIR='+constants.ROOT)
BATCH_SIZE=10
if __name__=='__main__':
    tf.reset_default_graph()

    dataset = tf.data.TFRecordDataset([constants.ROOT+'class.tfrecords'])
    dataset = dataset.map(cp.parser_fast).batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    image, caption = iterator.get_next()

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        sess.run(iterator.initializer)
        tf.global_variables_initializer().run()
        try:
            raw_img, lbl = sess.run([image, caption])
        except:
            sess.run(iterator.initializer)
        else:
            for i in range(BATCH_SIZE):
                plt.imshow(raw_img[i])
                plt.show()