import tensorflow as tf
import numpy as np
import json
import random


# Parameters
dimentions = (75, 75)
learning_rate = 0.001

# conv1
conv1_f = 5
conv1_num_filters = 32

conv2_f = 5
conv2_num_filters = 64


def load_data():
    # Read the data
    with open('data_train/processed/train.json') as data_file:
        data = json.load(data_file)

    random.shuffle(data)

    x_train = []
    y_train = []
    for i in range(len(data)-320):
        x_train.append(data[i]['band_1'])
        y_train.append([data[i]["is_iceberg"]])

    x_test = []
    y_test = []
    for i in range(len(data)-320, len(data)):
        x_test.append(data[i]['band_1'] + data[i]['band_2'])
        y_test.append([data[i]["is_iceberg"]])

    return x_train, x_test, y_train, y_test

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_tensor(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 75, 75, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([conv1_f, conv1_f, 1, conv1_num_filters])
        b_conv1 = bias_variable([conv1_num_filters])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([conv2_f, conv2_f, conv1_num_filters, conv2_num_filters])
        b_conv2 = bias_variable([conv2_num_filters])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    return h_pool2


"""
def fully_connected(conv1, conv2, angle1):

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([])
"""


def main():
    with tf.Session() as sess:

        t1 = tf.Variable([[1],[2],[3]])
        t2 = tf.Variable([[4],[5],[6]])
        t3 = tf.Variable([[7]]*3)

        sess.run(tf.initialize_all_variables())

        a = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]
        b = tf.concat([a,t3], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]
        print(sess.run(b))


if __name__ == '__main__':
    main()


