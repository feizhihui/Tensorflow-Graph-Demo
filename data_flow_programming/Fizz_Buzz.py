# encoding=utf-8

import tensorflow as tf


class FizzBuzz():
    def __init__(self, length=30):
        self.length = length  # 程序需要执行的序列长度
        self.array = tf.Variable([str(i) for i in range(1, length + 1)], dtype=tf.string, trainable=False)  # 最后程序返回的结果
        self.graph = tf.while_loop(self.cond, self.body, [1, self.array], )  # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _):
        return (tf.less(i, self.length + 1))  # 判断是否是最后一个值

    def body(self, i, _):
        flow = tf.cond(
            tf.equal(tf.mod(i, 15), 0),  # 如果值能被 15 整除，那么就把该位置赋值为 FizzBuzz
            lambda: tf.assign(self.array[i - 1], 'Fifteen'),

            lambda: tf.cond(tf.equal(tf.mod(i, 3), 0),  # 如果值能被 3 整除，那么就把该位置赋值为 Fizz
                            lambda: tf.assign(self.array[i - 1], 'Three'),
                            lambda: tf.cond(tf.equal(tf.mod(i, 5), 0),  # 如果值能被 5 整除，那么就把该位置赋值为 Buzz
                                            lambda: tf.assign(self.array[i - 1], 'Five'),
                                            lambda: self.array  # 最后返回的结果
                                            )
                            )
        )
        return (tf.add(i, 1), flow)


if __name__ == '__main__':
    fizzbuzz = FizzBuzz(length=50)
    ix, array = fizzbuzz.run()
    print(ix)
    print(array)
