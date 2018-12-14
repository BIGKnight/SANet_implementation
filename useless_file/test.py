import numpy as np
import tensorflow as tf
a = np.array([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
b = np.array([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
c = tf.constant([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
d = tf.constant([1., 2.])
# c = tf.divide(c, 5)
f = tf.subtract(c[0], d[0])
with tf.Session() as sess:
    print(sess.run(f))