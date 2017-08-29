import tensorflow as tf
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MINST_data/", one_hot=True)

sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+ b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    if i % 100 == 0:
        logging.info('train iteration {0}......'.format(i))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy after {0} iter is {1}'
              .format(i,accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0})))


