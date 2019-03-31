import tensorflow as tf
import os

print('work path', os.getcwd())

weights = tf.Variable([[2, 3], [0, 1]], dtype=tf.float32, name='weights')
bias = tf.Variable([[1, 3]], dtype=tf.float32, name='bias')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_path = saver.save(sess, 'My_net_test/save_temp.ckpt')
    print('Save to path:', saver_path)

