#this is a practice for mnist
from __future__ import  print_function
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


BATCH_SIZE = 100

LEARNING_RATE_BASE =  0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

CKPT_PATH = "/path/to/MNIST_data/"
CKPT_NAME = "model.ckpt"

def train(mnist):
	with tf.name_scope("input"):
		x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	y = mnist_inference.inference(x, regularizer)
	global_step = tf.Variable(0, trainable = False)
	
	with tf.name_scope("moving_average"):
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		variables_average_op = variable_averages.apply(tf.trainable_variables())

	with tf.name_scope("loss_function"):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		regularization = tf.add_n(tf.get_collection('losses'))
		loss = cross_entropy_mean + regularization

	with tf.name_scope("train_step"):
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			mnist.train.num_examples / BATCH_SIZE,
			LEARNING_RATE_DECAY)
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
		with tf.control_dependencies([train_step, variables_average_op]):
			train_op = tf.no_op(name = 'train')
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			op, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
			if i % 1000 ==0:
				print("After %d training steps, loss on training, loss is %g." % (step, loss_value))
				saver.save(sess, os.path.join(CKPT_PATH, CKPT_NAME), global_step = global_step)		
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict = {x: xs, y_: ys})
	writer = tf.summary.FileWriter("/path/to/log", tf.get_default_graph())


def main(argv=None):
	old_v = tf.logging.get_verbosity()
	tf.logging.set_verbosity(tf.logging.ERROR)
	mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot = True)
	train(mnist)
	tf.logging.set_verbosity(old_v)

if __name__ == '__main__':
	tf.app.run()

			
