import tensorflow as tf
import json
import random
import csv

# Parameters
dimentions = (50, 50)
learning_rate = 1e-4
keep_prob1 = 0.8
keep_prob2 = 0.8

batch_size = 15
num_iter = 1

# conv1
conv1_f = 5
conv1_num_filters = 15

# conv2
conv2_f = 5
conv2_num_filters = 30

num_hidden_fc1 = 1000
num_hidden_fc2 = 100
num_hidden_fc3 = 2


def load_data():
    # Read the data
    with open('data_train/processed/train.json') as data_file:
        data = json.load(data_file)

    random.shuffle(data)

    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    y_train = []

    for i in range(len(data) - 320):
        x_train_1.append(data[i]['band_1'])
        x_train_2.append(data[i]['band_2'])
        x_train_3.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_train.append([0, 1])
        else:
            y_train.append([1, 0])

    x_test_1 = []
    x_test_2 = []
    x_test_3 = []
    y_test = []
    for i in range(len(data) - 320, len(data)):
        x_test_1.append(data[i]['band_1'])
        x_test_2.append(data[i]['band_2'])
        x_test_3.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_test.append([0, 1])
        else:
            y_test.append([1, 0])

    return x_train_1, x_train_2, x_train_3, x_test_1, x_test_2, x_test_3, y_train, y_test

def load_test_data(file_name):
    # Read the data
    with open(file_name) as data_file:
        data = json.load(data_file)

    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    x_train_4 = []

    for i in range(len(data)):
        x_train_1.append(data[i]['band_1'])
        x_train_2.append(data[i]['band_2'])
        x_train_3.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        x_train_4.append(data[i]['id'])

    return x_train_1, x_train_2, x_train_3, x_train_4


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


def conv_tensor(x, W_conv1, b_conv1, W_conv2, b_conv2):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, dimentions[0], dimention[1], 1])

    with tf.name_scope('conv1'):
        # W_conv1 = weight_variable([conv1_f, conv1_f, 1, conv1_num_filters])
        # b_conv1 = bias_variable([conv1_num_filters])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        # W_conv2 = weight_variable([conv2_f, conv2_f, conv1_num_filters, conv2_num_filters])
        # b_conv2 = bias_variable([conv2_num_filters])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    return h_pool2


def fully_connected(conv1, conv2, angle):
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([2 * 19 * 19 * conv2_num_filters + 1, num_hidden_fc1])
        b_fc1 = bias_variable([num_hidden_fc1])
        h_pool2_conv1_flat = tf.reshape(conv1, [-1, 19 * 19 * conv2_num_filters])
        h_pool2_conv2_flat = tf.reshape(conv2, [-1, 19 * 19 * conv2_num_filters])

        temp_fc1 = tf.concat([h_pool2_conv1_flat, h_pool2_conv2_flat], 1)
        input_fc1 = tf.concat([temp_fc1, angle], 1)

        h_fc1 = tf.nn.relu(tf.matmul(input_fc1, W_fc1) + b_fc1)

    with tf.name_scope("dropout1"):
        kp_1 = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, kp_1)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([num_hidden_fc1, num_hidden_fc2])
        b_fc2 = bias_variable([num_hidden_fc2])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope("dropout2"):
        kp_2 = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, kp_2)

    with tf.name_scope("fc3"):
        W_fc3 = weight_variable([num_hidden_fc2, num_hidden_fc3])
        b_fc3 = bias_variable([num_hidden_fc3])

        pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return pred, kp_1, kp_2


def main():
    x_train_1, x_train_2, x_train_3, x_test_1, x_test_2, x_test_3, y_train, y_test = load_data()

    x1 = tf.placeholder(tf.float32, [None, dimentions[0] * dimentions[1]], name='x1')
    x2 = tf.placeholder(tf.float32, [None, dimentions[0] * dimentions[1]], name='x2')
    angle = tf.placeholder(tf.float32, [None, 1], name='angle')
    y = tf.placeholder(tf.float32, [None, 2], name="y")

    W_conv1 = weight_variable([conv1_f, conv1_f, 1, conv1_num_filters])
    b_conv1 = bias_variable([conv1_num_filters])

    W_conv2 = weight_variable([conv2_f, conv2_f, conv1_num_filters, conv2_num_filters])
    b_conv2 = bias_variable([conv2_num_filters])

    conv1_out = conv_tensor(x1, W_conv1, b_conv1, W_conv2, b_conv2)
    conv2_out = conv_tensor(x2, W_conv1, b_conv1, W_conv2, b_conv2)

    pred, keep1, keep2 = fully_connected(conv1_out, conv2_out, angle)

    softmax_pred = tf.nn.softmax(pred, name='output')

    cross_validtion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_validtion)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_iter):
            print(epoch)
            for i in range(0, len(x_train_1), batch_size):
                x_batch_1, x_batch_2, x_batch_3 = x_train_1[i:i + batch_size], x_train_2[i:i + batch_size], x_train_3[
                                                                                                            i:i + batch_size]
                if i + batch_size >= len(x_train_1):
                    x_batch_1, x_batch_2, x_batch_3 = x_train_1[i:], x_train_2[i:], x_train_3[i:]

                y_batch = y_train[i:i + batch_size]
                if i % (2 * batch_size) == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x1: x_batch_1, x2: x_batch_2, angle: x_batch_3, y: y_batch, keep1: 1.0,
                        keep2: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x1: x_batch_1, x2: x_batch_2, angle: x_batch_3, y: y_batch, keep1: keep_prob1,
                                          keep2: keep_prob2})

        print('test accuracy %g' % accuracy.eval(
            feed_dict={x1: x_test_1, x2: x_test_2, angle: x_test_3, y: y_test, keep1: 1.0, keep2: 1.0}))

        x_train_1, x_train_2, x_train_3, x_test_1, x_test_2, x_test_3, y_train, y_test = None,None,None,None,None,None,None,None

        file_names = ['test_processed_0_cropped.json','test_processed_1_cropped.json','test_processed_2_cropped.json','test_processed_3_cropped.json','test_processed_4_cropped.json','test_processed_5_cropped.json','test_processed_6_cropped.json','test_processed_7_cropped.json','test_processed_8_cropped.json','test_processed_9_cropped.json']
        with open('fries.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in (file_names):
                x_batch_1, x_batch_2, x_batch_3, x_batch_4 = load_test_data(i)
                out = sess.run(softmax_pred, feed_dict={x1: x_batch_1, x2: x_batch_2, angle: x_batch_3, keep1: 1.0, keep2: 1.0})
                for j in range(len(out)):
                    writer.writerow([x_batch_4[j],out[j][0]])
                print("file read")

if __name__ == '__main__':
    main()
