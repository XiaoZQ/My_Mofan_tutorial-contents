import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# creat tensorflow structre start #

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# creat tensorflow structre end #

sess = tf.Session()

sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

sess.close()

