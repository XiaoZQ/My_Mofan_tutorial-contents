import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(
    'D:\\statistical_software\\python_program\\My_Mofan_tutorial-contents\\MNIST_data\\',
    one_hot=True
)


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')
        biases = tf.Variable(tf.zeros([1, out_size]), name='biases')
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)

    return output


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(- tf.reduce_sum(ys * tf.log(prediction),
                                               reduction_indices=[1]))
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images, mnist.test.labels
            ))


# Terminal
# tensorboard --logdir=P22/logs/

# http://laptop-4l7t18js:6006/
