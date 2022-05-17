
import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 声明一个占位符, 运行计算时需要输入该值, 一般用作模型的输入,
#  None表示张量的第一个维度可以是任何长度
x = tf.placeholder("float", [None, 784])

# 声明可修改的张量, 可在计算时被修改, 一般用作模型的参数
# 声明时必须指定初始化值
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 声明模型
y = tf.nn.softmax(tf.matmul(x, w) + b)
# 声明目标输出
y_ = tf.placeholder("float", [None, 10])

# 声明交叉熵
# log(y)会计算y每个元素的对数,
# y_*tf.log(y)对应元素相乘
# reduce_sum计算所有元素和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 声明梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)

# 最小化交叉熵
train_step = optimizer.minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 循环训练1000次, 每次取100个样本进行训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, {x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))