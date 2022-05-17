#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
五子棋, 训练器实现

@author: CoderJie
@email: coderjie@outlook.com
"""

import random
import numpy as np


from collections import defaultdict, deque
from policy_value_net import PolicyValueNet
from gomoku_mcts import MCTSSelfPlayer, MCTSPlayer, rollout_policy_value
from gomoku_game import GomokuPlayer, GomokuBoard


class NetTrainer:
    """
    网络训练器
    """
    def __init__(self, init_model=None):
        # 棋盘宽度
        self.board_width = 11
        # 棋盘高度
        self.board_height = 11
        # 连子胜利数
        self.n_in_row = 5
        # 自我对弈次数
        self.self_game_num = 5000
        # 自奕指定次数后,检查棋力
        self.check_freq = 50
        # 重复训练次数
        self.repeat_train_epochs = 5
        # 训练时MCTS模拟次数
        self.train_play_out_n = 2000
        # 学习速度
        # self.learn_rate = 2e-3
        # 尝试修改学习速度
        self.learn_rate = 2e-4
        # 批量训练数据大小
        # self.batch_size = 500
        self.batch_size = 1000
        # 训练池最大大小
        self.buffer_max_size = 20000
        # 训练数据缓冲池
        self.data_buffer = deque(maxlen=self.buffer_max_size)

        # 对手MCTS模拟次数
        self.rival_play_out_n = 5000
        # 最佳胜率
        self.best_win_ratio = 0.0

        # 初始化策略网络
        self.brain = PolicyValueNet(self.board_width, self.board_height, init_model)
        # 初始化自我对弈玩家
        self.self_player = MCTSSelfPlayer(self.brain.policy_value, self.train_play_out_n)

    def self_play_once(self):
        """
        完成一次自我对弈, 收集训练数据
        :return: 训练数据
        """

        game_board = GomokuBoard(width=self.board_width,
                                 height=self.board_height,
                                 n_in_row=self.n_in_row)
        batch_states = []
        batch_probs = []
        current_players = []

        while True:
            action, acts_probs = self.self_player.get_action(game_board)
            # 保存当前状态
            batch_states.append(game_board.state())
            # 保存当前状态下进行各个动作的概率
            batch_probs.append(acts_probs)
            # 保存当前玩家
            current_players.append(game_board.current_player)

            # 执行动作
            game_board.move(action)

            # 检查游戏是否结束
            end, winner = game_board.check_winner()
            if end:
                batch_values = np.zeros(len(current_players))
                # 如果不是和局则将胜利者的状态值设置为1, 失败者的状态值设置为-1
                if winner == GomokuPlayer.Nobody:
                    batch_values[np.array(current_players) == winner] = 1.0
                    batch_values[np.array(current_players) != winner] = -1.0
                batch_values = np.reshape(batch_values, [-1, 1])
                return winner, list(zip(batch_states, batch_probs, batch_values))

    def get_equi_data(self, play_data):
        """
        获取等价数据(旋转和镜像)
        :param play_data:
        :return:
        """
        extend_data = []
        for state, prob, value in play_data:
            for i in [1, 2, 3, 4]:
                # 旋转数据
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_prob = np.rot90(prob.reshape(self.board_height, self.board_width), i)
                extend_data.append((equi_state, equi_prob.flatten(), value))

                # 左右镜像
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_prob = np.fliplr(equi_prob)
                extend_data.append((equi_state, equi_prob.flatten(), value))

        return extend_data

    def policy_evaluate(self, show=None):
        """
        棋力评估
        :return: 胜率
        """
        net_player = MCTSPlayer(self.brain.policy_value, self.train_play_out_n)
        mcts_player = MCTSPlayer(rollout_policy_value, self.rival_play_out_n)

        net_win = 0

        # 神经网络玩家执黑棋先手
        for i in range(5):
            game_board = GomokuBoard(width=self.board_width,
                                     height=self.board_height,
                                     n_in_row=self.n_in_row)
            while True:
                action = net_player.get_action(game_board)
                game_board.move(action)
                end, winner = game_board.check_winner()
                if show:
                    game_board.dbg_print()
                if end:
                    if show:
                        print(winner)
                    if winner == GomokuPlayer.Black:
                        net_win += 1
                    break

                action = mcts_player.get_action(game_board)
                game_board.move(action)
                end, winner = game_board.check_winner()
                if show:
                    game_board.dbg_print()
                if end:
                    if show:
                        print(winner)
                    break

        # MCTS玩家执黑棋先手
        for i in range(5):
            game_board = GomokuBoard(width=self.board_width,
                                     height=self.board_height,
                                     n_in_row=self.n_in_row)
            while True:
                action = mcts_player.get_action(game_board)
                game_board.move(action)
                end, winner = game_board.check_winner()
                if show:
                    game_board.dbg_print()
                if end:
                    if show:
                        print(winner)
                    break

                action = net_player.get_action(game_board)
                game_board.move(action)
                end, winner = game_board.check_winner()
                if show:
                    game_board.dbg_print()
                if end:
                    if show:
                        print(winner)
                    if winner == GomokuPlayer.White:
                        net_win += 1
                    break

        return net_win/10

    def run(self):
        """
        开始训练
        """
        black_win_num = 0
        white_win_num = 0
        nobody_win_num = 0

        for i in range(self.self_game_num):

            # 收集训练数据
            print("Self Game: {}".format(i))
            winner, play_data = self.self_play_once()
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            if winner == GomokuPlayer.Black:
                black_win_num += 1
            elif winner == GomokuPlayer.White:
                white_win_num += 1
            else:
                nobody_win_num += 1
            print("Black: {:.2f} White: {:.2f} Nobody: {:.2f}".format(
                  black_win_num/(i+1),
                  white_win_num/(i+1),
                  nobody_win_num/(i+1)))

            # 积累一些数据后, 进行训练
            if len(self.data_buffer) > (self.batch_size * 2):
                mini_batch = random.sample(self.data_buffer, self.batch_size)

                batch_states = [data[0] for data in mini_batch]
                batch_probs = [data[1] for data in mini_batch]
                batch_values = [data[2] for data in mini_batch]

                total_loss = 0.0
                total_entropy = 0.0
                for j in range(self.repeat_train_epochs):
                    loss, entropy = self.brain.train(batch_states,
                                                     batch_probs,
                                                     batch_values,
                                                     self.learn_rate)
                    total_loss += loss
                    total_entropy += entropy
                print("Loss: {:.2f}, Entropy: {:.2f}".format(total_loss/self.repeat_train_epochs,
                                                             total_entropy/self.repeat_train_epochs))

            if (i+1) % self.check_freq == 0:
                self.brain.save_model(".\\CurrentModel\\GomokuAi")
                win_ratio = self.policy_evaluate()
                print("Rival({}), Net Win Ratio: {:.2f}".format(self.rival_play_out_n, win_ratio))
                if win_ratio > self.best_win_ratio:
                    self.best_win_ratio = win_ratio
                    self.brain.save_model(".\\BestModel\\GomokuAi")
                if self.best_win_ratio >= 1.0:
                    self.best_win_ratio = 0.0
                    self.rival_play_out_n += 1000


if __name__ == '__main__':
    trainer = NetTrainer(".\\CurrentModel\\GomokuAi")
    trainer.run()
