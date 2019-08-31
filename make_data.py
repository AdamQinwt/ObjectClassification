import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import constants

def make(record_name='class.tfrecords',num_pics=-1,):
    if num_pics<0: num_pics=constants.IMG_NUM
    writer = tf.python_io.TFRecordWriter(record_name)
    with open(constants.ANS_DIR,'r') as ans:
        lines=ans.readlines()
        classes={'nothing':0}
        for indx in range(0,num_pics):
            lbl=lines[indx+1].split(',')[1].strip('\n')
            img=cv2.imread(constants.IMG_DIR+str(indx+1)+'.png').reshape([-1])/255.0
            try:
                cls=classes[lbl]
            except:
                classes[lbl]=len(classes)
                cls=classes[lbl]
            l=np.zeros([constants.CLASS_NUM],dtype=np.float)
            l[cls-1]=1.0
            example = tf.train.Example(features=tf.train.Features(feature=
                                                                {"label":
                                                                    tf.train.Feature(
                                                                        float_list=tf.train.FloatList(
                                                                            value=l)),
                                                                    "features":
                                                                        tf.train.Feature(
                                                                            float_list=tf.train.FloatList(
                                                                                value=img)),
                                                                }))
            s = example.SerializeToString()
            writer.write(s)
            print(str(indx+1)+' written.')
        #lst_classes=classes.keys()
        with open("labels.txt", 'w') as ans:
            for c in classes:
                ans.write(c+':'+str(classes[c]))
                print(c+':'+str(classes[c]))
    writer.close()

if __name__=='__main__':
    make(num_pics=100)