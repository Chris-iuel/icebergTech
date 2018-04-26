
import tensorflow as tf
import random
import json


with open("C:\\Users\\Christopher\\Desktop\\icebergTECH\\data_train\\processed\\train.json") as f:
   data = json.load(f)
   data_train = data[:1000]
   data_test = data[1000:]


### CNN CONFIG


n_classes = 1
batch_size = 50

x = tf.placeholder('float', [None, 5625])  # 75*75 = 5625   28*28 = 784
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([32 * 32 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 75,75, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 19 * 19 * 24])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def nextbatch(x,data):
	data_subset = [[],[]]
	sample = random.sample(range(len(data)), x)
	print("---------------",len(sample))
	for i in sample:
		data_subset[0].append(data[i]["band_1"])
		data_subset[1].append([data[i]["is_iceberg"]])
		#print (data_subset[0])
		#print("len",len(data_subset[0]))
		#_=input()
	return data_subset


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.square(prediction - y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #saver
    saver = tf.train.Saver()
    return optimizer, cost


hm_epochs = 50
with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(500):
                epoch_data = nextbatch(hm_epochs,data_train)
                epoch_x = epoch_data[0]
                epoch_y = epoch_data[1]
		
                _, c = sess.run([train_neural_network(epoch_x)], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            #print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            print("t")



#train_neural_network(data[1])