import tensorflow as tf
std_dev=0.001
def inference(x,train=False):

    conv1_1 = tf.layers.conv2d(inputs=x,
                               filters=32,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               data_format='channels_last',
                               activation=tf.nn.relu,
                               use_bias=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                               bias_initializer=tf.constant_initializer(0.0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               trainable=train,
                               name='conv1_1'
                               )
    conv1_2 = tf.layers.conv2d(inputs=conv1_1,
                               filters=64,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               data_format='channels_last',
                               activation=tf.nn.relu,
                               use_bias=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                               bias_initializer=tf.constant_initializer(0.0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               trainable=train,
                                 name='conv1_2'
                               )
    conv1_3 = tf.layers.conv2d(inputs=conv1_2,
                               filters=128,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               data_format='channels_last',
                               activation=tf.nn.relu,
                               use_bias=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                               bias_initializer=tf.constant_initializer(0.0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               trainable=train,
                                 name='conv1_3'
                               )
    conv1_4 = tf.layers.conv2d(inputs=conv1_3,
                               filters=256,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               data_format='channels_last',
                               activation=tf.nn.relu,
                               use_bias=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                               bias_initializer=tf.constant_initializer(0.0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               trainable=train,
                               name='conv1_4'
                               )
    conv1_5 = tf.layers.conv2d(inputs=conv1_4,
                               filters=512,
                               kernel_size=5,
                               strides=2,
                               padding='SAME',
                               data_format='channels_last',
                               activation=tf.nn.relu,
                               use_bias=True,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                               bias_initializer=tf.constant_initializer(0.0),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               bias_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               trainable=train,
                               name='conv1_5'
                               )
    # reshape
    shape = conv1_5.get_shape().as_list()
    print(shape)
    nodes = shape[1] * shape[2] * shape[3]
    print('nodes=' + str(nodes))
    reshaped = tf.reshape(conv1_5, [-1, nodes])
    bn1=tf.layers.batch_normalization(inputs=reshaped,training=train)
    fc_score_1=tf.layers.dense(inputs=bn1,
                        units=120,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                        bias_initializer=tf.constant_initializer(0.0),
                        trainable=train,
                               name='fc1'
                        )
    fc_score_1 = tf.layers.dropout(fc_score_1, training=train)
    fc_score_3 = tf.layers.dense(inputs=fc_score_1,
                                 units=10,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                 bias_initializer=tf.constant_initializer(0.0),
                                 trainable=train,
                               name='fc2'
                                 )
    variables = tf.contrib.framework.get_variables_to_restore()
    return tf.nn.softmax(fc_score_3),variables