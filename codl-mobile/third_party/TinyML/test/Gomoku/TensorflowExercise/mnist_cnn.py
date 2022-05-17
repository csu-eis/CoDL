
import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    # 卷积函数
    #
    # x 做卷积的输入图像, 要求是一个Tensor,
    # 具有[batch, height, width, channels]这样的shape[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    # w 卷积核, 要求是一个Tensor
    # 具有[height, width, in_channels, out_channels]这样的shape[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    # strides 卷积时在图像每一维的步长, 对于图片，因为只有两维，通常strides取[1，stride，stride，1]
    # padding 只能是"SAME","VALID", "SAME"进行边界0填充, "VALID"不进行边界填充
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # 最大化池化
    # x 输入 具有shape[batch, height, width, channels]
    # ksize 池化窗口大小一般为[1, height, width, 1]
    # strides 滑动步长
    # padding 填充模式
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 声明一个占位符, 运行计算时需要输入该值, 一般用作模型的输入,
#  None表示张量的第一个维度可以是任何长度
x_2d = tf.placeholder("float", [None, 784])
# 把x变为4维向量, -1代表自动计算该维度大小[batch, height, width, channels]
x_4d = tf.reshape(x_2d, [-1, 28, 28, 1])

# 声明5x5, 输入通道数为1, 输出通道数为32的卷积, 也就是32个卷积核
w_conv1 = weight_variable([5, 5, 1, 32])
# 声明卷积核的偏置
b_conv1 = bias_variable([32])

# 第一层卷积, 输入为[?, 28, 28, 1], 输出为[?, 28, 28, 32], 使用斜坡激活函数
h_conv1 = tf.nn.relu(conv2d(x_4d, w_conv1) + b_conv1)
# 进行最大化池化, 输入[?, 28, 28, 32], 输出[?, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 声明5x5, 输入通道数为32, 输出通道数为64的卷积, 也就是64个卷积核
w_conv2 = weight_variable([5, 5, 32, 64])
# 声明卷积核的偏置
b_conv2 = bias_variable([64])

# 第二层卷积, 输入为[?, 14, 14, 32], 输出为[?, 14, 14, 64], 使用斜坡激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 进行最大化池化, 输入[?, 14, 14, 64], 输出[?, 7, 7, 64]
h_pool2 = max_pool_2x2(h_conv2)

# 加入1024个神经元的全连接层
# 声明全连接层的权重
w_fc1 = weight_variable([7*7*64, 1024])
# 声明全连接层的偏置
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 全连接层输出[?, 1024]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层, 输出[?, 10]
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 声明目标输出
y_ = tf.placeholder("float", [None, 10])


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
with sess.as_default():
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x_2d: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x_2d: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x_2d: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))