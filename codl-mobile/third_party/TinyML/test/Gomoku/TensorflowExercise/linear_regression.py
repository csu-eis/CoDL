
import tensorflow as tf
import numpy as np


# 随机生成二维数组2*100的随机数[0, 1)
x_data = np.float32(np.random.rand(2, 100))

# 生成目标值, 矩阵积y=w*x+b
y_data = np.dot([0.100, 0.200], x_data) + 0.003

# 定义变量b
b = tf.Variable(tf.zeros([1]))

# 定义变量权重w, 1*2, [-1.0, 1.0) 
w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

# 定义目标变量y
y = tf.matmul(w, x_data) + b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 定义优化算法: 梯度下降, 学习速度0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化Variable
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))
        





