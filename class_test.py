import tensorflow as tf
import constants
import os
import sys
import inference
import class_parser as cp
import numpy as np
import platform
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('system='+platform.system())
print('DIR='+constants.ROOT)
BATCH_SIZE=128
if __name__=='__main__':
    tf.reset_default_graph()
    pic=tf.placeholder(dtype=tf.float32,shape=[None,constants.IMG_SHAPE[0],constants.IMG_SHAPE[1],constants.IMG_SHAPE[2]])
    score_=tf.placeholder(dtype=tf.float32,shape=[None,constants.CLASS_NUM])
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    global_step = tf.Variable(initial_value=0, trainable=False)

    score,variables=inference.inference(pic,False)

    dataset = tf.data.TFRecordDataset([constants.ROOT+'class.tfrecords'])
    dataset = dataset.map(cp.parser_fast).batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    image, caption = iterator.get_next()

    with tf.Session(config=config) as sess:
        tf.summary.FileWriter('logs/',sess.graph)
        tf.local_variables_initializer().run()
        sess.run(iterator.initializer)
        saver = tf.train.Saver(var_list=variables)
        saver_all=tf.train.Saver()
        tf.global_variables_initializer().run()
        try:
            saver.restore(sess, constants.ROOT + inference.MODEL_NAME+'.ckpt')
        except Exception as e:
            print(str(e))
            print('Cannot restore!')
        cnt = 0.0
        total = 0.0
        while True:
            try:
                raw_img, lbl = sess.run([image, caption])
            except:
                sess.run(iterator.initializer)
                #print("After %d training step(s), loss on training batch is %g." % (epoch, total / cnt))
                print('Correct:\t'+str(cnt))
                print('Total:\t'+str(total))
                print('Accuracy:\t'+str(cnt/total*100)+'%')
                break
            else:
                scores = sess.run(score, feed_dict={pic: raw_img})
                for i in range(BATCH_SIZE):
                    try:
                        print('scores='+str(scores[i]))
                        print('answers='+str(lbl[i]))
                        if np.argmax(scores[i])==np.argmax(lbl[i]):
                            cnt+= 1
                        total += 1
                    except:
                        break