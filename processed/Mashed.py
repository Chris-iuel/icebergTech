import tensorflow as tf
import json
import random
import csv
import numpy
import matplotlib.pyplot as plt

file_names = ['test_processed_0_cropped.json',
              'test_processed_1_cropped.json',
              'test_processed_2_cropped.json',
              'test_processed_3_cropped.json',
              'test_processed_4_cropped.json',
              'test_processed_5_cropped.json',
              'test_processed_6_cropped.json',
              'test_processed_7_cropped.json',
              'test_processed_8_cropped.json',
              'test_processed_9_cropped.json']

# Parameters
dev_set_size = 320
batch_size = 50
dimentions = (50, 50)

learning_rate = 8e-4
learning_rate_dec = 2e-5
learning_rate_min = 1e-4
keep_prob1 = 1
keep_prob2 = 1

l2 = 0.010

num_iter = 1000

conv_layers = 9
conv_f = [3]*conv_layers
conv_num_filters = [10]*conv_layers
num_blocks = 2

channels = 1
createCSV = False


num_hidden_fc1 = 75
num_hidden_fc2 = 10
num_hidden_fc3 = 2


def load_data():
    # Read the data
    with open('C:\\Users\\Christopher\\Desktop\\icebergTECH\\processed\\train_processed_cropped.json') as data_file:
        data = json.load(data_file)

    random.shuffle(data)

    x_train_1 = []
    x_train_2 = []
    x_train_angle = []
    y_train = []

    for i in range(len(data) - dev_set_size):
        x_train_1.append(data[i]['band_1'])
        x_train_2.append(data[i]['band_2'])
        x_train_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_train.append([0, 1])
        else:
            y_train.append([1, 0])

    x_test_1 = []
    x_test_2 = []
    x_test_angle = []
    y_test = []
    for i in range(len(data) - dev_set_size, len(data)):
        x_test_1.append(data[i]['band_1'])
        x_test_2.append(data[i]['band_2'])
        x_test_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_test.append([0, 1])
        else:
            y_test.append([1, 0])

    return x_train_1, x_train_2, x_train_angle, x_test_1, x_test_2, x_test_angle, y_train, y_test

def load_test_data(file_name):
    # Read the data
    with open('C:\\Users\\Christopher\\Desktop\\icebergTECH\\processed\\'+file_name) as data_file:
        data = json.load(data_file)

    x_test_1 = []
    x_test_2 = []
    x_test_angle = []
    id = []

    for i in range(len(data)):
        x_test_1.append(data[i]['band_1'])
        x_test_2.append(data[i]['band_2'])
        x_test_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        id.append(data[i]['id'])

    return x_test_1, x_test_2, x_test_angle, id

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


def conv_calc(conv_input,W_conv,b_conv):
    return tf.nn.relu(conv2d(conv_input, W_conv) + b_conv)


def conv_tensor(x,conv_list):

    with tf.name_scope('reshape'):
        conv_input = tf.reshape(x, [-1, 50, 50, channels])
    for i in range(len(conv_list)):
        conv_input = conv_calc(conv_input,conv_list[i][0],conv_list[i][1])
        if (i+1)%int(conv_layers/num_blocks) == 0:
            conv_input = (max_pool_2x2(conv_input))

    return conv_input

def conv_wb_create(conv_f,prev_conv_num_filters,conv_num_filters):

    W_conv = weight_variable([conv_f, conv_f, prev_conv_num_filters, conv_num_filters])
    b_conv = bias_variable([conv_num_filters])
    return [W_conv,b_conv]

def conv_create(conv_layers,conv_f,conv_num_filters) :
    conv_list = []
    conv_list.append(conv_wb_create(conv_f[0],1, conv_num_filters[0]))
    for i in range(1, conv_layers):
        conv_list.append(conv_wb_create(conv_f[i], conv_num_filters[i-1], conv_num_filters[i]))
    return conv_list

def conv_run(h_pool,w_conv,b_conv):
    return tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)



def fully_connected(conv1, conv2, angle):
    with tf.name_scope("fc1"):
        #TODO dimention (7) error because we dont max pool

        fc_dim0 = int(numpy.ceil(dimentions[0]/int(2**num_blocks)))
        fc_dim1 = int(numpy.ceil(dimentions[1]/int(2**num_blocks)))
        #pluss one for angle
        W_fc1 = weight_variable([2 * fc_dim0 * fc_dim1 * conv_num_filters[-1] + 1, num_hidden_fc1])
        b_fc1 = bias_variable([num_hidden_fc1])
        h_pool2_conv1_flat = tf.reshape(conv1, [-1, fc_dim0 * fc_dim1 * conv_num_filters[-1]])
        h_pool2_conv2_flat = tf.reshape(conv2, [-1, fc_dim0 * fc_dim1 * conv_num_filters[-1]])

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

    return pred, kp_1, kp_2, W_fc1, W_fc2, W_fc3


def main():
    x_train_1, x_train_2, x_train_angle, x_test_1, x_test_2, x_test_angle, y_train, y_test = load_data()

    x1 = tf.placeholder(tf.float32, [None, dimentions[0] * dimentions[1]], name='x1')
    x2 = tf.placeholder(tf.float32, [None, dimentions[0] * dimentions[1]], name='x2')
    x_angle = tf.placeholder(tf.float32, [None, 1], name='angle')
    y = tf.placeholder(tf.float32, [None, 2], name="y")

    conv_list = conv_create(conv_layers,conv_f, conv_num_filters)

    conv1_out = conv_tensor(x1, conv_list)
    conv2_out = conv_tensor(x2, conv_list)


    pred, keep1, keep2, W_fc1, W_fc2, W_fc3 = fully_connected(conv1_out, conv2_out, x_angle)

    softmax_pred = tf.nn.softmax(pred, name='output')



    cross_validtion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    #TODO Check implementation

    loss_sum = 0

    for i in conv_list:
        loss_sum += tf.nn.l2_loss(i[0])

    regularization = l2 * ((loss_sum) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3))

    loss = tf.reduce_mean(cross_validtion + regularization)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        test_plot = []

        for epoch in range(num_iter):
            epoch_avg = 0
            print(epoch)
            for i in range(0, len(x_train_1), batch_size):
                x_batch_1, x_batch_2, x_batch_4 = x_train_1[i:i + batch_size], x_train_2[i:i + batch_size], x_train_angle[i:i+batch_size]
                y_batch = y_train[i:i + batch_size]
                if i + batch_size >= len(x_train_1):
                    x_batch_1, x_batch_2, x_batch_4 = x_train_1[i:], x_train_2[i:],x_train_angle[i:]
                    y_batch = y_train[i:]

                if i % (2 * batch_size) == 0:
                    train_accuracy = accuracy.eval(feed_dict={x1: x_batch_1, x2: x_batch_2, x_angle: x_batch_4, y: y_batch, keep1: 1.0,keep2: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    epoch_avg += train_accuracy
                train_step.run(feed_dict={x1: x_batch_1, x2: x_batch_2, x_angle: x_batch_4, y: y_batch, keep1: keep_prob1, keep2: keep_prob2})
            epoch_avg = (epoch_avg / (len(x_train_1) / batch_size / 2))
            test_plot.append(epoch_avg)
            print(epoch_avg)
        if dev_set_size != 0:
            test_val = accuracy.eval(
                feed_dict={x1: x_test_1, x2: x_test_2, x_angle: x_test_angle, y: y_test, keep1: 1.0, keep2: 1.0})
            print('test accuracy %g' % test_val)

        #if learning_rate > learning_rate_min:
        #   learning_rate -= learning_rate_dec

        x_test_1 = x_test_2 = x_test_angle = x_train_1 = x_train_2 = x_train_angle = None

        plt.plot(test_plot)
        plt.axhline(test_val, color='r')
        plt.show()

        if createCSV == True :
            with open('fries(DELETE).csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'is_iceberg'])
                for i in (file_names):
                    x_train_1, x_train_2, angle, id = load_test_data(i)
                    for j in range(0, len(x_train_1), batch_size):
                        x_batch_1, x_batch_2, x_batch_angle, id_batch = x_train_1[j:j + batch_size], \
                                                                                   x_train_2[j:j + batch_size], \
                                                                                   angle[j:j + batch_size], \
                                                                                   id[j:j + batch_size]
                        if j + batch_size >= len(x_train_1):
                            x_batch_1, x_batch_2, x_batch_angle, id_batch = x_train_1[j:],x_train_2[j:], angle[j:],id[j:]

                        out = sess.run(softmax_pred,
                                       feed_dict={x1: x_batch_1, x2: x_batch_2, x_angle: x_batch_angle,
                                                  keep1: 1.0, keep2: 1.0})
                        for k in range(len(out)):
                            writer.writerow([id_batch[k], out[k][0]])
                        x_batch_1 = x_batch_2 = x_batch_angle = id_batch = None
                    x_train_1 = x_train_2 = angle = id = None
                    print("file_read:", i)



if __name__ == '__main__':
    main()
