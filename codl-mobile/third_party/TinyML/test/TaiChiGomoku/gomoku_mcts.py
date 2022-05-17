#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
五子棋蒙特卡洛树搜索实现

@author: CoderJie
@email: coderjie@outlook.com
"""

import copy
import random
import math
import numpy as np

from gomoku_game import GomokuPlayer
from operator import itemgetter


def rollout_policy_value(board):
    """
    随机rollout策略值
    :param board: 棋盘
    :return:动作概率, 以及状态值
    """
    # 针对每个动作计算平均概率
    avg_probs = [1.0 / len(board.avl_actions) for i in range(len(board.avl_actions))]
    actions_probs = list(zip(board.avl_actions, avg_probs))

    state_value = 0.0
    # 随机进行动作计算状态值
    current_player = board.current_player
    for i in range(board.width * board.height):
        # rand_probs = np.random.rand(len(state.avl_actions))
        # 给各个动作随机出不同的概率
        rand_probs = [random.random() for j in range(len(board.avl_actions))]
        rand_action_probs = zip(board.avl_actions, rand_probs)

        # 找出概率值最大的动作
        max_action = max(rand_action_probs, key=itemgetter(1))[0]
        # 进行动作
        board.move(max_action)
        # 判断游戏是否结束, 以及胜利者
        end, winner = board.check_winner()
        if end:
            if winner == current_player:
                state_value = 1.0
            elif winner == GomokuPlayer.Nobody:
                state_value = 0.0
            else:
                state_value = -1.0
            break

    return actions_probs, state_value


class MCTSNode(object):
    """
    蒙特卡洛树搜索节点
    """

    def __init__(self, parent, prob, c_uct=5.0):
        """
        构造函数
        :param parent: 父节点
        :param prob: 节点先验概率
        :param c_uct: 上限置信区间参数
        """
        self.__parent = parent          # 父节点
        self.__children = {}            # 子节点字典, 键: 动作 值: 节点

        self.__c_uct = c_uct            # 上限置信区间参数

        self.__action_prob = prob       # 动作先验概率
        self.__action_q = 0.0           # 动作值即动作价值
        self.__action_n = 0             # 动作访问次数

    def expand(self, actions_prob):
        """
        扩展
        :param actions_prob: 动作先验概率列表
        """
        for action, prob in actions_prob:
            if action not in self.__children:
                self.__children[action] = MCTSNode(self, prob)

    def select(self):
        """
        选择一个动作
        :return: 返回(动作, 节点)元组
        """
        return max(self.__children.items(),
                   key=lambda action_node: action_node[1].__get_value())

    def backup(self, leaf_value):
        """
        反向更新节点
        :param leaf_value: 叶子节点(终局)评估值
        """
        # 更新父节点, 父节点和当前节点评估值相反
        if self.__parent:
            self.__parent.backup(-leaf_value)
        self.__update(leaf_value)

    def is_leaf(self):
        """
        是否是叶子节点
        :return: 叶子节点返回True, 否正返回False
        """
        return self.__children == {}

    def is_root(self):
        """
        是否是根节点
        :return: 根节点返回True, 否正返回False
        """
        return self.__parent is None

    def get_action_count(self):
        """
        获取动作进行次数
        :return: 动作进行次数
        """
        return self.__action_n

    def get_children(self):
        """
        获取子节点
        :return: 字节的字典, 键: 动作 值: 节点
        """
        return self.__children

    def set_parent(self, node):
        """
        设置父节点
        """
        self.__parent = node

    def __update(self, leaf_value):
        """
        更新节点, 更新动作值为模拟过程的平均值
        :param leaf_value: 叶子节点(终局)评估值
        """
        sum_value = self.__action_q * self.__action_n + leaf_value
        # 访问次数+1
        self.__action_n += 1
        # 更新动作值
        self.__action_q = sum_value/self.__action_n

    def __get_value(self):
        """
        获取调整后的动作值
        :return: 动作值
        """
        # 上限置信区间参数 用于平衡深度优先搜索和广度优先搜索, 该值越大会越倾向广度优先搜索
        u = self.__c_uct * self.__action_prob * math.sqrt(self.__parent.__action_n) / (1 + self.__action_n)
        return self.__action_q + u

    def __str__(self):
        if self.__parent is None:
            return "MCTSNode: Prob: {}, Q: {}, N: {}".format(
                self.__action_prob, self.__action_q, self.__action_n)
        else:
            return "MCTSNode: Prob: {}, Q: {}, N: {} Value: {}".format(
                self.__action_prob, self.__action_q, self.__action_n, self.__get_value())


class MCTS:
    """
    蒙特卡洛树搜索
    """
    def __init__(self, policy_value_fn, play_out_n):
        """
        构造函数
        :param policy_value_fn: 策略值函数
        :param play_out_n: 模拟次数
        """
        # 根节点
        self.__root = MCTSNode(None, 1.0)
        self.__policy_value = policy_value_fn
        self.__play_out_n = play_out_n

    def get_action_probs(self, board):
        """
        根据当前棋盘进行蒙特卡洛树搜索, 获取进行下个动作概率
        :param board: 棋盘
        :return:
        """

        # 进行多次模拟
        for i in range(self.__play_out_n):
            board_copy = copy.deepcopy(board)
            self.__play_out(board_copy)

        # 获取每个动作进行的次数
        children = self.__root.get_children().items()
        actions_counts = [(action, child.get_action_count()) for action, child in children]
        # 解压数据
        actions, counts = zip(*actions_counts)

        # 访问次数加上1, 避免访问次数为0, 得到概率为0的状况
        counts = tuple(map(lambda count: count+1, counts))
        sum_count = sum(counts)
        # 计算每个动作的概率, 概率正比于该动作被进行的次数
        probs = tuple(map(lambda count: count/sum_count, counts))

        return actions, probs

    def update_action(self, action):
        """
        保留上次模拟经验, 更新下一动作
        """
        children = self.__root.get_children()
        if children == {}:
            return

        if action in children:
            self.__root = children[action]
            self.__root.set_parent(None)
        else:
            raise Exception('Error action: {}'.format(action))

    def reset(self):
        """
        重置蒙特卡洛树搜索
        """
        self.__root = MCTSNode(None, 1.0)

    def __play_out(self, board):
        """
        进行一次模拟
        :param board: 棋盘, 本方法会改动棋盘进行模拟
        """
        # 根据以往模拟进行动作, 直到新区域
        current_node = self.__root
        while True:
            if current_node.is_leaf():
                break
            action, current_node = current_node.select()
            board.move(action)

        # 检查游戏是否结束以及胜利者
        end, winner = board.check_winner()

        if not end:
            # 获取动作概率, 以及状态值
            actions_probs, state_value = self.__policy_value(board)
            current_node.expand(actions_probs)
            # 状态值表明的是当前状态下当前玩家的优劣状况
            # 进入该状态的动作由前一玩家进行, 所以动作值取反
            current_node.backup(-state_value)
        else:
            # 游戏和棋结束
            if winner == GomokuPlayer.Nobody:
                current_node.backup(0)
            else:
                # 游戏不是和棋结束, 则代表当前状态下当前玩家输棋, 所以进入该状态的玩家的动作值设为1
                current_node.backup(1)


class MCTSPlayer:
    """
    纯蒙特卡洛树搜索玩家
    """
    def __init__(self, policy_value_fn, play_out_n=1000):
        """
        构造函数
        :param play_out_n: 模拟次数
        """
        self.__mcts = MCTS(policy_value_fn, play_out_n)

    def get_action(self, board):
        """
        获取动作
        """
        # 如果游戏已经结束则返回None
        end, winner = board.check_winner()
        if end:
            return None

        acts, probs = self.__mcts.get_action_probs(board)
        acts_probs = zip(acts, probs)

        # 找出最大概率的动作
        action = max(acts_probs, key=itemgetter(1))[0]
        self.__mcts.reset()

        return action

    def get_action_prob(self, board):
        """
        获取动作
        """
        # 如果游戏已经结束则返回None
        end, winner = board.check_winner()
        if end:
            return None

        acts, probs = self.__mcts.get_action_probs(board)
        acts_probs = zip(acts, probs)

        # 找出最大概率的动作
        action_prob = max(acts_probs, key=itemgetter(1))
        self.__mcts.reset()

        return action_prob


class MCTSSelfPlayer:
    """
    蒙特卡洛树搜索自弈玩家(用于生成训练数据)
    """
    def __init__(self, policy_value_fn, play_out_n=1000):
        """
        构造函数
        :param play_out_n: 模拟次数
        """
        self.__mcts = MCTS(policy_value_fn, play_out_n)

    def get_action(self, board):
        """
        获取动作
        """
        # 如果游戏已经结束则返回None
        end, winner = board.check_winner()
        if end:
            return None

        acts, probs = self.__mcts.get_action_probs(board)
        self.__mcts.reset()

        # 蒙特卡洛树搜索返回的是可执行动作的概率
        # 作为训练数据, 我们需要把不可执行的动作的概率值设置为0
        all_acts_probs = np.zeros(board.height*board.width)
        all_acts_probs[list(acts)] = probs

        # 为了提高训练数据的多样性, 我们需要在概率值上增加噪声
        # 在概率分布上选择一个动作
        noise = np.random.dirichlet(0.3 * np.ones(len(probs)))
        action = np.random.choice(acts, p=0.75 * np.array(probs) + 0.25 * noise)

        return action, all_acts_probs


def print_node(node, space='', action=None):
    print(space, action, node)
    for action, child in node.get_children().items():
        print_node(child, space + '  ', action)


