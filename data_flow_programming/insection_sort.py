# encoding=utf-8


import numpy as np
import tensorflow as tf

"""
x = np.array([4, 3, 2, 1])
for i in range(len(x)):
    for j in range(i, 0, -1):
        if x[j - 1] > x[j]:
            t = x[j - 1]
            x[j - 1] = x[j]
            x[j] = t
        else:
            break
"""


class InsectionSort():
    def __init__(self, array):
        length = len(array)
        self.array = tf.Variable(array, trainable=False)
        cond = lambda i, _: tf.less(i, length)
        i = tf.constant(0)
        # for(i=0; i<len; outer_loop(i,arr));
        self.graph = tf.while_loop(cond, self.outer_loop, loop_vars=[i, self.array], parallel_iterations=1)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            graph = sess.run(self.graph)
            return graph

    def outer_loop(self, i, array):
        j = tf.identity(i)
        cond = lambda j, _: tf.greater(j, 0)
        # for(j=i; j>0; inner_loop(j,arr));
        loop = tf.while_loop(cond, self.inner_loop, loop_vars=[j, array], parallel_iterations=1)
        return tf.add(i, 1), loop[1]

    def inner_loop(self, j, array):
        def replace():
            a = self.array[j]
            b = self.array[j - 1]
            return tf.scatter_nd_update(self.array, [[j - 1], [j]], [a, b])

        body = tf.cond(tf.greater(self.array[j - 1], self.array[j]),
                       # update tensor by replace
                       lambda: tf.scatter_nd_update(self.array, [[j - 1], [j]], [self.array[j], self.array[j - 1]]),
                       # replace,
                       lambda: array)
        return tf.subtract(j, 1), body


if __name__ == '__main__':
    x = np.array([4, 3, 2, 1, 4, 3, 2, 1])  # 1., 7., 3., 8.
    insection = InsectionSort(x)
    _, arr = insection.run()
    print(x)
    print(arr)  # return a random result ?

    y = np.random.rand(20)
    print(y)
    _, sorted_array = InsectionSort(y).run()
    print(sorted_array)
