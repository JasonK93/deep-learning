"""
This is the first try in the tensorflow,
which is used to make a simple model to recognize dight
with a brief softmax regression

"""
import logging
import tensorflow as tf

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
# load the mnist dataset
logging.info('Step1 : Load Data ...')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MINST_data/", one_hot=True)

logging.info('Step 2 ： check the size of data')
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784], name='digit_feature')
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10], name='digit_label')

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        logging.info('train iteration {0}......'.format(i))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy after {0} iter is {1}'
              .format(i,accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


