import tensorflow as tf

state = tf.Variable(0, name='counter')
# print(state.name)

one = tf.constant(1)

new_data = tf.add(state, one)
updata = tf.assign(state, new_data)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(updata)
        print(sess.run(state))

