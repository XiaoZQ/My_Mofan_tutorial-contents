import tensorflow as tf


with tf.Session() as sess:
    a = tf.constant(123, tf.float64)
    print(sess.run(a))

