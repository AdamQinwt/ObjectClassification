import platform
if platform.system()=='Linux':
    ROOT='/home/sdd/qinwentao/classification/train/'
    ANS_DIR='/home/sdd/qinwentao/classification/train/trainLabels.csv'
    IMG_DIR='/home/sdd/qinwentao/classification/train/train/'
else:
    ROOT='E:\\Code\\Python\\ObjectClassification\\'
    IMG_DIR = 'E:\\datasets\\train\\train\\'
    ANS_DIR='E:\\datasets\\train\\trainLabels.csv'
IMG_SHAPE=[32,32,3]
IMG_SIZE=IMG_SHAPE[0]*IMG_SHAPE[1]*IMG_SHAPE[2]
IMG_NUM=50000
CLASS_NUM=10