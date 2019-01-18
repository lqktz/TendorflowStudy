# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)

# 定义批次大小
batch_size = 100
n_batch = mnist.train.num_examples

# 定义placeholder
x = tf.placeholder(tf.float32, [1, 784], name='input_x')
y = tf.placeholder(tf.float32, [1, 10], name='output_y')

# 定义 测试
x_test = tf.placeholder(tf.float32, [None, 784], name='input_test_x')
y_test = tf.placeholder(tf.float32, [None, 10], name='input_test_y')

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([1, 10]), name="b")

prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 创建损失函数
train = tf.train.GradientDescentOptimizer(0.02).minimize(tf.reduce_mean(tf.square(y - prediction)))


# 名称转换
def canonical_name(x):
    return x.name.split(":")[0]


# 计算准确率
test_prediction = tf.nn.softmax(tf.matmul(x_test, W) + b)
accuarcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, 1), tf.argmax(test_prediction, 1)), tf.float32))

init = tf.global_variables_initializer()
out = tf.identity(prediction, name="output")

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            for index in range(len(batch_xs)):
                xs = batch_xs[index].reshape(1, 784)
                ys = batch_ys[index].reshape(1, 10)
                sess.run(train, feed_dict={x: xs, y: ys})

        acc = sess.run(accuarcy, feed_dict={x_test: mnist.test.images, y_test: mnist.test.labels})
        print("over" + str(acc))

    frozen_tensors = [out]
    out_tensors = [out]

    frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   list(map(canonical_name, frozen_tensors)))
    tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [x], out_tensors)

    open("writer_model.tflite", "wb").write(tflite_model)
