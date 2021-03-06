# encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

np.random.seed(123)
"""
for i in range(len(x)):
    for j in range(len(x) - 1, i, -1):
        if x[j - 1] > x[j]:
            t = x[j - 1]
            x[j - 1] = x[j]
            x[j] = t
"""


class BubbleSort():
    def __init__(self, array):
        i = tf.constant(0)
        self.length = len(array)

        cond = lambda i, _: tf.less(i, self.length)
        self.array = tf.Variable(array, trainable=False)
        self.graph = tf.while_loop(cond, self.outer_loop, loop_vars=[i, self.array],
                                   shape_invariants=[i.get_shape(),
                                                     tf.TensorShape(self.length)],
                                   parallel_iterations=1,
                                   )  # back_prop=False

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def outer_loop(self, i, _):
        j = tf.constant(self.length - 1)
        cond = lambda i, j, _: tf.greater(j, i)
        loop = tf.while_loop(cond, self.inner_loop, loop_vars=[i, j, self.array],
                             shape_invariants=[i.get_shape(), j.get_shape(), tf.TensorShape(self.length)],
                             parallel_iterations=1, back_prop=False)
        return tf.add(i, 1), loop[2]

    def inner_loop(self, i, j, _):
        # array must be a mutable tensor
        body = tf.cond(tf.greater(self.array[j - 1], self.array[j]),
                       lambda: tf.scatter_nd_update(self.array, [[j - 1], [j]], [self.array[j], self.array[j - 1]]),
                       lambda: self.array)
        return i, tf.subtract(j, 1), body


if __name__ == '__main__':
    x = np.array([1., 7., 3., 8., 1., 7., 3., 8.])
    _, sorted_array = BubbleSort(x).run()
    print(x)
    print(sorted_array)

    y = np.random.rand(20)
    print(y)
    _, sorted_array = BubbleSort(y).run()
    print(sorted_array)
