import tensorflow as tf


with tf.Session() as sess:
    a = tf.constant('1234')
    print(sess.run(a))

