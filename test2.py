from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random



#NOTE
#cut down on file size or feature maps.
#enable 2 features (band 1 and 2)
#add angle to convolutional decition network
#add filters that learn?
#make semi simeze/2channel network architecture
# train each channel on a diff computer??
# train filters on test data?
#cut out background noise to allow more feature maps with smaller pics what is new pic size?



import argparse
import sys
import tempfile

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 5625), where 5625 is the
    number of pixels in each scan.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 1), with values
    equal to the logits of classifying the digit into one class (is_iceberg). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.	----------------CHANGE THIS TO 2 TO ALLOW BAND_2 ???
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 75, 75, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps. 
  # REDUCED IT FROM 32 TO 12 FEATUREMAPS
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 12])
    b_conv1 = bias_variable([12])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 12, 24])
    b_conv2 = bias_variable([24])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 75x75 image
  # is down to (75/2/2) 18???*18*64 = 20736 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([19 * 19 * 24, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 19*19*24])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 1 class
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


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

def nextbatch(x,data):
	data_subset = [[],[]]
	sample = random.sample(range(len(data)), x)
	for i in sample:
		data_subset[0].append(data[i]["band_1"])
		data_subset[1].append([data[i]["is_iceberg"]])
		#print (data_subset[0])
		#print("len",len(data_subset[0]))
		#_=input()
	return data_subset


def main(_):
  # Import data
  with open("C:\\Users\\Christopher\\Desktop\\icebergTECH\\data_train\\processed\\train.json") as f:
   data = json.load(f)
   data_train = data[:1000]
   data_test = data[1000:] 

  # Create the model
  x = tf.placeholder(tf.float32, [None, 5625])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  #with tf.name_scope('loss'):
    #----------------cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.reduce_mean(tf.square(y_ - y_conv)))

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    #formats test data
    data_test = nextbatch(len(data_test),data_test)
    sess.run(tf.global_variables_initializer())
    for i in range(500):
      batch = nextbatch(50,data_train)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
 
    print('test accuracy %g' % accuracy.eval(feed_dict={x: data_test[0], y_: data_test[1], keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)