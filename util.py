import numpy as np
import selective_search as ss
import cv2
import copy
#preprocesses including input, padding, COLOR_RGB2BGR, COLOR_RGB2GRAY
def preprocess(filename,padX=0,padY=0,RGB2BGR=False,GRAY=False,new_size=(0,0)):
    if GRAY:
        img = cv2.imread(filename,0)
    else:
        img=cv2.imread(filename)
    if RGB2BGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if padX>0 or padY>0:
        img = np.pad(array=img, pad_width=[padX, padY], mode='constant')
    if new_size[0]!=0 and new_size[1]!=0:
        img=cv2.resize(img,new_size,0,0)
    return img

#calculate intersect of two boxes
#boxes are formatted as [scores,x,y,w,h]
def get_intersect(b1,b2,num_scores=0):
    if b1[num_scores]<b2[num_scores]:
        x1=b1[num_scores]
        x2=b2[num_scores]
        w1=b1[num_scores+2]
        w2=b2[num_scores+2]
    else:
        x1 = b2[num_scores]
        x2 = b1[num_scores]
        w1 = b2[num_scores+2]
        w2 = b1[num_scores+2]
    dw=x1-x2+w1
    if dw<0: return 0.0
    elif dw>w2: dw=w2

    if b1[num_scores+1] < b2[num_scores+1]:
        y1 = b1[num_scores+1]
        y2 = b2[num_scores+1]
        h1 = b1[num_scores+3]
        h2 = b2[num_scores+3]
    else:
        y1 = b2[num_scores+1]
        y2 = b1[num_scores+1]
        h1 = b2[num_scores+3]
        h2 = b1[num_scores+3]
    dh = y1 - y2 + h1
    if dh < 0:
        return 0.0
    elif dh > h2:
        dh = h2
    return dw*dh

#calculate iou of two boxes
#boxes are formatted as [scores,x,y,w,h]
def get_iou(b1,b2,num_scores=0):
    intersect=get_intersect(b1,b2,num_scores)
    return intersect/(b1[num_scores+2]*b1[num_scores+3]+b2[num_scores+2]*b2[num_scores+3]-intersect)

#find the box in a list of boxes
#boxes are formatted as [scores,x,y,w,h]
#returns the max iou and the index of the matching box
def find_in_boxes(b,boxes,num_scores=0):
    max_indx=0
    max_iou=get_iou(b,boxes[0],num_scores)
    for i in range(1,len(boxes)):
        iou=get_iou(b,boxes[i],num_scores)
        if iou==1.0:
            max_indx = i
            max_iou = 1.0
            break
        elif iou>max_iou:
            max_indx=i
            max_iou=iou
    return max_iou,max_indx

#nms
#boxes are formatted as [scores,x,y,w,h]
def nms(boxes,iou_thresh=0.9,num_scores=0):
    arr = np.vstack(boxes).reshape(-1, 5)
    scores = arr[:, 0]
    inds = np.argsort(scores)[::-1]
    lst = arr[inds]
    accept = []
    l = len(lst)
    # print(l)
    active = np.ones([l])
    for i in range(l):
        if active[i] == 0: continue
        # print(str(lst[i])+"accepted")
        j = i + 1
        while j < l:
            iou=get_iou(lst[i],lst[j],True)
            # print("t="+str(t)+"\tarea="+str(area))
            if iou > iou_thresh:
                active[i]+=1
                active[j] = 0
            j += 1
    for i in range(l):
        if active[i]<1: continue
        else:
            accept.append(lst[i])
    return accept

#input image should be formatted as RGB
#returns a list of bounding boxes [scores,x,y,w,h]
def selective_search_areas(img,scale=500, sigma=0.9, min_size=10,ratio=-1,min_area=1000,num_scores=0):
    # Workaround to import selectivesearch from its directory
    # Perform selective search
    boxes=[]
    img_lbl, regions = ss.selective_search(
        img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = set()
    for r in regions:
        # Exclude same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # Exclude small regions
        if r['size'] < min_area:
            continue
        # Exclude distorted rects
        min_x, min_y, max_x, max_y = r['rect']
        if min_x>max_x: min_x,max_x=max_x,min_x
        if min_x>max_x: min_y,max_y=max_y,min_y
        width, height = max_x - min_x + 1, max_y - min_y + 1
        #print('width=' + str(width) + '\theight=' + str(height))
        if width == 0 or height == 0:
            continue
        if ratio>0:
            if width / height > ratio or height / width > ratio:
                continue
        candidates.add(r['rect'])
    for min_x, min_y, max_x, max_y in candidates:
        width, height = max_x - min_x + 1, max_y - min_y + 1
        boxes.append([1.0 for i in range(num_scores)]+[min_x,min_y,width,height])
    return boxes

def save_img(img,filename='img.bmp'):
    cv2.imwrite(filename, img)

#boxes are [scores,x,y,w,h]
def draw_areas(img,boxes,num_scores=0,color=(255,0,0),thickness=2):
    img_show=copy.deepcopy(img)
    for b in boxes:
        cv2.rectangle(img_show,(int(b[num_scores]),int(b[num_scores+1])),(int(b[num_scores])+int(b[num_scores+2]),int(b[num_scores+1])+int(b[num_scores+3])),color=color,thickness=thickness)
    return img_show

#boxes are [class,x,y,w,h]
def draw_areas_on_classes(img,boxes,colors,thickness):
    img_show=copy.deepcopy(img)
    for b in boxes:
        cv2.rectangle(img_show,(int(b[1]),int(b[2])),(int(b[1])+int(b[3]),int(b[2])+int(b[4])),color=colors[int(b[0])],thickness=thickness[int(b[0])])
    return img_show

def img_to_list(img):
    return list(img.reshape([-1]))

def print_list(lst):
    for e in lst:
        print(str(e))

#make an array of length that total such that arr[i]=0.0 for i != num and arr[num]=1.0
def num_to_array(num,total):
    arr=np.zeros([total],dtype=np.float)
    arr[num]=1.0
    return arr