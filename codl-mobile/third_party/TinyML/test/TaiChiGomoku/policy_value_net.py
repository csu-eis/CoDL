#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略值网络实现
依赖TensorFlow 1.10.0版本

@author: CoderJie
@email: coderjie@outlook.com
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet:
    """
    策略值网络
    """
    def __init__(self, board_width, board_height, model_file=None):
        """
        构造函数
        :param board_width: 棋盘宽度
        :param board_height: 棋盘高度
        """

        self.__board_width = board_width
        self.__board_height = board_height

        # 定义模型输入占位符, 张量形状为[?, 4, height, width]
        # None表示张量的第一维为任意长度, 第一维即代表样本数量, 就是表示模型接收任意的样本数,
        # 并且每个样本包含4个 board_height*board_width 的表格
        self.__input_data = tf.placeholder(tf.float32,
                                           shape=[None, 4, board_height, board_width])

        # 定义训练所用的状态值占位符
        self.__train_value = tf.placeholder(tf.float32,
                                            shape=[None, 1])

        # 定义训练所用的动作概率值占位符
        self.__train_prob = tf.placeholder(tf.float32,
                                           shape=[None, board_height * board_width])

        # 定义学习速度占位符
        self.__learning_rate = tf.placeholder(tf.float32)

        # 组建神经网络
        self.__build_net()

        # 创建一个会话, 并进行初始化动作
        self.__session = tf.Session()
        self.__session.run(tf.global_variables_initializer())

        # 保存器, 用于保存或者恢复模型
        self.__saver = tf.train.Saver()

        # 从文件中恢复模型
        if model_file is not None:
            self.restore_model(model_file)

    def __del__(self):
        """
        析构函数
        """
        self.__session.close()

    def train(self, batch_states, batch_probs, batch_values, lr):
        """
        训练模型
        :param batch_states: 批量状态(形状为[?, 4, height, width])
        :param batch_probs: 批量概率(形状为[?, height*width])
        :param batch_values: 批量状态值(形状为[?, 1])
        :param lr: 学习速度(浮点数)
        :return: 损失值, 熵
        """
        loss, entropy, _ = self.__session.run(
                [self.__total_loss, self.__entropy, self.__optimizer],
                feed_dict={self.__input_data: batch_states,
                           self.__train_prob: batch_probs,
                           self.__train_value: batch_values,
                           self.__learning_rate: lr})
        return loss, entropy

    def policy_value(self, board):
        """
        获取策略(动作概率), 状态值
        :param board: 棋盘对象
        :return: 策略(动作概率), 状态值
        """
        state = board.state()
        input_data = state.reshape(-1, 4, self.__board_height, self.__board_width)
        acts_prob, state_value = self.__policy_value(input_data)

        # 只获取有效动作的概率
        avl_acts = board.avl_actions
        acts_prob = list(zip(avl_acts, acts_prob[0][avl_acts]))
        return acts_prob, state_value[0][0]

    def save_model(self, path):
        """
        保存模型
        :param path: 模型路径
        """
        self.__saver.save(self.__session, path)

    def restore_model(self, path):
        """
        恢复模型
        :param path: 模型路径
        """
        self.__saver.restore(self.__session, path)

    def __build_net(self):
        """
        组建卷积神经网络
        :return:
        """
        # 转置输入数据, 输出张量形状[?, height, width, 4]
        # [0, 2, 3, 1] 0 代表原始数据第一维, 1 代表原始数据第二维, 2 代表原始数据第三维, 3 代表原始数据第四维
        # 转置后可以把棋盘数据看作是通道数为4, 大小为height*width的图片
        input_trans = tf.transpose(self.__input_data, [0, 2, 3, 1])

        # 第一层卷积层, 32个卷积核, 输出张量形状[?, height, width, 32]
        # 卷积核数量32, 卷积核大小 3*3, 卷积核的通道数就是输入数据的通道数4, 卷积步长默认为1
        # 填充方式为"same" 表示边界填充0
        # channels_last表示通道为最后一维
        # 默认加上偏置, 偏置初始值默认为0
        # 激活函数选择斜坡函数
        conv1 = tf.layers.conv2d(inputs=input_trans,
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 data_format="channels_last",
                                 activation=tf.nn.relu)

        # 第二层卷积层, 64个卷积核, 输出张量形状[?, height, width, 64]
        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 data_format="channels_last",
                                 activation=tf.nn.relu)

        # 第三层卷积层, 128个卷积核, 输出张量形状[?, height, width, 128]
        conv3 = tf.layers.conv2d(inputs=conv2,
                                 filters=128,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 data_format="channels_last",
                                 activation=tf.nn.relu)

        # 策略端降维, 输出张量形状[?, height, width, 4]
        policy_conv = tf.layers.conv2d(inputs=conv3,
                                       filters=4,
                                       kernel_size=[1, 1],
                                       padding="same",
                                       data_format="channels_last",
                                       activation=tf.nn.relu)

        # 策略端修改数据维度, 输出张量形状[?, 4 * height * width]
        policy_flat = tf.reshape(policy_conv,
                                 [-1, 4 * self.__board_height * self.__board_width])

        # 策略端全连接层, 输出张量形状[?, height * width]
        # units代表输出大小
        # 默认加上偏置, 偏置初始值默认为0
        # 使用log_softmax激活函数
        self.__policy_fc = tf.layers.dense(inputs=policy_flat,
                                           units=self.__board_height * self.__board_width,
                                           activation=tf.nn.log_softmax)

        # 价值端降维, 输出张量形状[?, height, width, 2]
        value_conv = tf.layers.conv2d(inputs=conv3,
                                      filters=2,
                                      kernel_size=[1, 1],
                                      padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)

        # 价值端修改数据维度, 输出张量形状[?, 2 * height * width]
        value_flat = tf.reshape(value_conv,
                                [-1, 2 * self.__board_height * self.__board_width])

        # 价值端64个神经元的全连接层, 输出张量形状[?, 64]
        # 使用斜坡激活函数
        value_fc1 = tf.layers.dense(inputs=value_flat,
                                    units=64,
                                    activation=tf.nn.relu)

        # 价值端1个神经元的全连接层, 输出张量形状[?, 1]
        # 使用双曲正切激活函数, 双曲正切函数的值域是(-1, 1)
        self.__value_fc2 = tf.layers.dense(inputs=value_fc1,
                                           units=1,
                                           activation=tf.nn.tanh)

        # 定义状态值损失函数(平方差)
        value_loss = tf.losses.mean_squared_error(self.__train_value, self.__value_fc2)

        # 定义策略损失函数
        policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(
            tf.multiply(self.__train_prob, self.__policy_fc), 1)))

        # l2正则化系数
        l2_beta = 1e-4
        # 获取所有可训练变量
        vs = tf.trainable_variables()
        # 定义正则项
        l2_penalty = l2_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vs if 'bias' not in v.name.lower()])

        # 定义总损失函数
        self.__total_loss = value_loss + policy_loss + l2_penalty

        # 定义优化器
        self.__optimizer = tf.train.AdamOptimizer(
            learning_rate=self.__learning_rate).minimize(self.__total_loss)

        # 计算策略的熵值, 用于监控网络
        self.__entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.__policy_fc) * self.__policy_fc, 1)))

    def __policy_value(self, states):
        """
        获取策略(动作概率), 状态值
        :param states: 批量状态二值数据, 数据形状应该为[?, 4, height, width]
        :return: 策略(形状: [?, height*width]), 状态值(形状: [?, 1])
        """
        log_probs, values = self.__session.run([self.__policy_fc, self.__value_fc2],
                                               feed_dict={self.__input_data: states})
        acts_probs = np.exp(log_probs)
        return acts_probs, values

