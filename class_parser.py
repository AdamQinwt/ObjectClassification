import tensorflow as tf
import constants
def parser_fast(serialized_example):
    context, sequence = tf.parse_single_sequence_example(serialized_example,
                                                         context_features={
                                                             'features': tf.FixedLenFeature([constants.IMG_SIZE],
                                                                                            tf.float32),
                                                             'label': tf.FixedLenFeature(
                                                                 [constants.CLASS_NUM], tf.float32)
                                                         })
    feat = context['features']
    feat=tf.reshape(feat,constants.IMG_SHAPE)
    label = context['label']
    label = tf.reshape(label, [constants.CLASS_NUM])
    return feat, label