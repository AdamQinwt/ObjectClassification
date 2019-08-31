import tensorflow as tf
import constants
import os
import sys
#import inference
import inference_complex as inference
import class_parser as cp
import platform
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('system='+platform.system())
print('DIR='+constants.ROOT)
BATCH_SIZE=128
if __name__=='__main__':
    tf.reset_default_graph()
    LEARNING_RATE_BASE = float(sys.argv[1])
    steps=int(sys.argv[2])
    learning_rate=LEARNING_RATE_BASE
    pic=tf.placeholder(dtype=tf.float32,shape=[None,constants.IMG_SHAPE[0],constants.IMG_SHAPE[1],constants.IMG_SHAPE[2]])
    score_=tf.placeholder(dtype=tf.float32,shape=[None,constants.CLASS_NUM])
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    global_step = tf.Variable(initial_value=0, trainable=False)

    score,variables=inference.inference(pic,True)
    loss_score = -tf.reduce_mean(
        score_ * tf.log(tf.clip_by_value(score, 1e-10, 1.0))
        + (1 - score_) * tf.log(tf.clip_by_value(1 - score, 1e-10, 1.0))
    )
    loss=loss_score

    dataset = tf.data.TFRecordDataset([constants.ROOT+'class.tfrecords'])
    dataset = dataset.map(cp.parser_fast).batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    image, caption = iterator.get_next()

    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
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
        epoch = 0
        while epoch < steps:
            cnt = 0
            total = 0.0
            while True:
                try:
                    raw_img, lbl = sess.run([image, caption])
                except:
                    sess.run(iterator.initializer)
                    #print("After %d training step(s), loss on training batch is %g." % (epoch, total / cnt))
                    print(str(epoch)+'\t'+str(total/cnt))
                    epoch = epoch + 1
                    break
                else:
                    # print(lbl)
                    sess.run(train_step,
                             feed_dict={pic: raw_img, score_: lbl})
                    losses = sess.run(loss, feed_dict={pic: raw_img, score_: lbl})
                    total = total + losses
                    cnt = cnt + 1
        saver_all.save(sess, constants.ROOT + inference.MODEL_NAME+'.ckpt')