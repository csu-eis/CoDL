#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
五子棋, 棋盘实现

@author: CoderJie
@email: coderjie@outlook.com
"""

import numpy as np
from enum import Enum, unique


@unique
class GomokuPlayer(Enum):
    Nobody = -1  # 无子
    Black = 0    # 黑子
    White = 1    # 白子


class GomokuBoard(object):
    """
    五子棋棋盘
    """

    def __init__(self, **kwargs):
        """
        构造函数
        :param kwargs: 关键字参数
        """
        # 棋盘宽度
        self.width = int(kwargs.get('width', 9))
        # 棋盘高度
        self.height = int(kwargs.get('height', 9))
        # 胜利连子数
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # 最近动作
        self.last_action = -1
        # 黑子先手
        self.current_player = GomokuPlayer.Black
        # 可用的动作
        self.avl_actions = list(range(self.width * self.height))
        # 当前棋盘
        self.board = [[GomokuPlayer.Nobody for col in range(self.width)] for row in range(self.height)]
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('GomokuBoard width and height can not be '
                            'less than {}'.format(self.n_in_row))

    def state(self):
        """
        获取当前局面的二值描述
        :return: 数据形状: [4, height, width]
        """

        # 使用4个height*width的二值特征平面来描述局面
        # 第一个平面是当前玩家的棋子位置, 有棋子的位置是1, 没棋子的位置是0
        # 第二个平面是对手玩家的棋子位置, 有棋子的位置是1, 没棋子的位置是0
        # 第三个平面是对手玩家最近的落子位置, 整个平面只有一个位置是1, 其余全是0
        # 第四个平面表示当前玩家是否是先手玩家, 是则整个平面全为1, 否正全为0

        square_state = np.zeros((4, self.height, self.width))

        if self.current_player == GomokuPlayer.Black:
            riva_player = GomokuPlayer.White
        else:
            riva_player = GomokuPlayer.Black

        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] == self.current_player:
                    square_state[0][row][col] = 1.0
                elif self.board[row][col] == riva_player:
                    square_state[1][row][col] = 1.0

        if self.last_action != -1:
            row, col = self.action2loc(self.last_action)
            square_state[2][row][col] = 1.0

        if self.current_player == GomokuPlayer.Black:
            square_state[3][:, :] = 1.0

        return square_state

    def move(self, action):
        """
        下子
        :param action: 动作值
        """
        if action not in self.avl_actions:
            raise Exception('GomokuBoard move action error ')

        row, col = self.action2loc(action)
        self.board[row][col] = self.current_player
        self.avl_actions.remove(action)
        self.last_action = action

        # 改变当前玩家
        if self.current_player == GomokuPlayer.Black:
            self.current_player = GomokuPlayer.White
        else:
            self.current_player = GomokuPlayer.Black

    def check_winner(self):
        """
        检查胜利者
        :return: (游戏是否结束, 赢家)
        """
        if self.last_action == -1:
            return False, GomokuPlayer.Nobody

        act_row, act_col = self.action2loc(self.last_action)
        last_player = self.board[act_row][act_col]

        # 检查横向是否达成连子
        spot_count = 1
        for col in range(act_col-1, -1, -1):
            if self.board[act_row][col] == last_player:
                spot_count += 1
            else:
                break
        for col in range(act_col + 1, self.width):
            if self.board[act_row][col] == last_player:
                spot_count += 1
            else:
                break
        if spot_count >= self.n_in_row:
            return True, last_player

        # 检查纵向是否达成连子
        spot_count = 1
        for row in range(act_row-1, -1, -1):
            if self.board[row][act_col] == last_player:
                spot_count += 1
            else:
                break
        for row in range(act_row + 1, self.height):
            if self.board[row][act_col] == last_player:
                spot_count += 1
            else:
                break
        if spot_count >= self.n_in_row:
            return True, last_player

        # 检查斜线是否达成连子
        spot_count = 1
        row = act_row-1
        col = act_col-1
        while row >= 0 and col >= 0:
            if self.board[row][col] == last_player:
                spot_count += 1
            else:
                break
            row -= 1
            col -= 1
        row = act_row+1
        col = act_col+1
        while row < self.height and col < self.width:
            if self.board[row][col] == last_player:
                spot_count += 1
            else:
                break
            row += 1
            col += 1
        if spot_count >= self.n_in_row:
            return True, last_player

        # 检查反斜线是否达成连子
        spot_count = 1
        row = act_row+1
        col = act_col-1
        while row < self.height and col >= 0:
            if self.board[row][col] == last_player:
                spot_count += 1
            else:
                break
            row += 1
            col -= 1
        row = act_row-1
        col = act_col+1
        while row >= 0 and col < self.width:
            if self.board[row][col] == last_player:
                spot_count += 1
            else:
                break
            row -= 1
            col += 1
        if spot_count >= self.n_in_row:
            return True, last_player

        if len(self.avl_actions) == 0:
            return True, GomokuPlayer.Nobody

        return False, GomokuPlayer.Nobody

    def action2loc(self, action):
        """
        动作值转换为位置
        3*3 棋盘如下:
        0 1 2
        3 4 5
        6 7 8
        动作5的位置是(1, 2)
        :param action: 动作值
        :return: 位置元组(row, col)
        """
        if action not in range(self.width * self.height):
            raise Exception('Action error: {}'.format(action))

        row = action // self.width
        col = action % self.width
        return row, col

    def loc2action(self, loc):
        """
        位置转换为动作值
        :param loc: 位置
        :return: 动作
        """
        if len(loc) != 2:
            raise Exception('Location error: {}'.format(loc))
        row = loc[0]
        col = loc[1]
        action = row * self.width + col

        if action not in range(self.width * self.height):
            raise Exception('Location error: {}'.format(loc))
        return action

    def dbg_print(self):
        print('')
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row][col] == GomokuPlayer.Nobody:
                    print(' - ', end='')
                if self.board[row][col] == GomokuPlayer.Black:
                    print(' X ', end='')
                if self.board[row][col] == GomokuPlayer.White:
                    print(' O ', end='')
            print('')
        print('')
