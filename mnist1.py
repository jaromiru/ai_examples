# MNIST recognition implementation with TensorFlow
# Linear Regression
# ------------------------------------------------

import tensorflow as tf

# download data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# prepare variables
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_uniform([784, 10], -0.1, 0.1))
b = tf.Variable(tf.zeros([10]))

# constants
l = 0.1
m = tf.cast( tf.size(y) / 10, tf.float32 )

# hypotesis & cost function
a = tf.sigmoid( tf.matmul(x, W) + b )

c = -(y * tf.log(a + 1e-10)) - ((1-y) * tf.log(1 - a - 1e-10))
reg = l / 2.0 / m * tf.reduce_sum(W * W)

loss = tf.reduce_mean( tf.reduce_sum( c, reduction_indices=[1] )) + reg 

# debug
correct = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 

# setup training
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()


# init session
sess = tf.Session()
sess.run(init)

# batch_xs = mnist.train.images;	#use whole training set
# batch_ys = mnist.train.labels;
# train
for step in range(100000):	
	batch_xs, batch_ys = mnist.train.next_batch(100)	#use stochastic GD
	sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

	if step % 1000 == 0:	
		test_a = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
		train_a = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
		print()
		print(step, ": test_accuracy  = ", test_a)
		print("    train_accuracy = ", train_a)