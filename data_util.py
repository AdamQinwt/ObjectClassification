import random
import util
#get random boxes like [scores,x,y,w,h] in an area
def get_random_box(max_x=0,max_y=0,min_x=0,min_y=0,num_scores=0):
    box=[]
    for c in range(num_scores):
        box.append(0.0)
    x1=min_x+(max_x-min_x)*random.randint(1,10000)/10000.0
    y1=min_y+(max_y-min_y)*random.randint(1,10000)/10000.0
    w = (max_x - x1) * random.randint(1, 10000) / 10000.0
    h = (max_y - y1) * random.randint(1, 10000) / 10000.0
    box.extend([x1,y1,w,h])
    return box

#answers look like [class,x,y,w,h]
#new boxes are [class,x,y,w,h]
def get_random_boxes(answers,number=100,max_x=0,max_y=0,min_x=0,min_y=0,iou_thresh=0.8):
    boxes=[]
    for i in range(number):
        box=get_random_box(max_x,max_y,min_x,min_y,1)
        max_iou,max_indx=util.find_in_boxes(box,answers,1)
        if max_iou>iou_thresh:
            boxes.append([answers[max_indx][0],box[1],box[2],box[3],box[4]])
        else:
            boxes.append([0, box[1], box[2], box[3], box[4]])
    return boxes