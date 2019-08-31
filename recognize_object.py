import cv2
import matplotlib.pyplot as plt
import util
import constants
import numpy as np
import tensorflow as tf
import inference
import copy

colors=[(255,255,255),
        (255,255,0),
        (0, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 0, 255),
        (127, 0, 0),
        (0, 127, 0),
        (0, 0, 127),
        (127, 127, 0)
        ]
# read dict
def read_labels(file_name='labels.txt'):
    classes = {}
    with open(file_name, 'r') as ans:
        lines = ans.readlines()
        for line in lines:
            s=line.strip('\n').split(':')
            classes[int(s[1])]=s[0]
    return classes


# generate proposals from selective search
def get_proposals(file_name,num_scores=0):
    img=util.preprocess(file_name)
    boxes=util.selective_search_areas(img,num_scores=num_scores)
    return img,boxes

# resize regions in the boxes[scores,x,y,w,h]
def resize_boxes(img,bboxes,num_scores=0):
    regions=np.zeros(shape=[len(bboxes)]+constants.IMG_SHAPE)
    indx=0
    for box in bboxes:
        try:
            print('box='+str(box))
            region=img[box[num_scores+1]:box[num_scores+1]+box[num_scores+3],box[num_scores]:box[num_scores]+box[num_scores+2],:]
            print(region.shape)
            region=cv2.resize(region,(constants.IMG_SHAPE[1],constants.IMG_SHAPE[0]),interpolation=cv2.INTER_CUBIC)
            regions[indx]=region/255.0
            indx+=1
        except:
            continue
    return regions,len(bboxes)

# run classification
def run_classification(regions,bboxes,num_regions=1,class_thresh=0.5,num_scores=0):
    tf.reset_default_graph()
    pic = tf.constant(regions,dtype=tf.float32)
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    score, variables = inference.inference(pic, False)
    obj=[]
    with tf.Session(config=config) as sess:
        #tf.summary.FileWriter('logs/',sess.graph)
        tf.local_variables_initializer().run()
        saver = tf.train.Saver(var_list=variables)
        tf.global_variables_initializer().run()
        try:
            saver.restore(sess, constants.ROOT +'classification.ckpt')
        except Exception as e:
            print(str(e))
            print('Cannot restore!')
        scores = sess.run(score)
        for i in range(num_regions):
            try:
                print('scores='+str(scores[i]))
                #print('answers='+str(lbl[i]))
                cls=np.argmax(scores[i])
                if scores[i,cls]<class_thresh: continue#obj.append([0,0.0]+bboxes[i][num_scores:])
                else: obj.append([cls+1,scores[i,cls]]+bboxes[i][num_scores:])
            except:
                break
    return obj

# show classification results
def show_results(file_name,items,colors,labels):
    img_show=util.preprocess(file_name,RGB2BGR=False)
    for r in items:
        #print(r)
        img_show=cv2.rectangle(img_show,(r[2],r[3]),(r[2]+r[4],r[3]+r[5]),color=colors[r[0]],thickness=1)
        font=cv2.FONT_HERSHEY_SIMPLEX
        img_show=cv2.putText(img_show,labels[r[0]],(r[2],r[3]),font,0.6,colors[r[0]],2)
        img_show=cv2.putText(img_show,str(r[1]),(r[2]+r[4],r[3]+r[5]),font,0.3,colors[r[0]],2)
    plt.imshow(img_show)
    plt.show()
    return img_show

if __name__=='__main__':
    IMG_NAME = 'Laysan_Albatross_0047_619.jpg'
    labels=read_labels()
    img,boxes=get_proposals(IMG_NAME)
    regions,num=resize_boxes(img,boxes)
    o=run_classification(regions,boxes,num,0.99)
    im_show=show_results(IMG_NAME,o,colors,labels)
    util.save_img(im_show,'show.bmp')