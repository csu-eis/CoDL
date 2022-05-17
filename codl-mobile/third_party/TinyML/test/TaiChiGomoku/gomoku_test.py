#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试五子棋

@author: CoderJie
@email: coderjie@outlook.com
"""

from policy_value_net import PolicyValueNet
from gomoku_mcts import MCTSPlayer, rollout_policy_value
from gomoku_game import GomokuPlayer, GomokuBoard


def net_self_play(board_width, board_height, n_in_row):
    game_board = GomokuBoard(width=board_width,
                             height=board_height,
                             n_in_row=n_in_row)
    brain = PolicyValueNet(board_width, board_height, ".\\CurrentModel\\GomokuAi")
    net_player = MCTSPlayer(brain.policy_value, 2000)

    while True:
        action = net_player.get_action(game_board)
        game_board.move(action)
        end, winner = game_board.check_winner()
        game_board.dbg_print()
        if end:
            print(winner)
            break


def net_mcts_play(board_width, board_height, n_in_row):
    game_board = GomokuBoard(width=board_width,
                             height=board_height,
                             n_in_row=n_in_row)
    brain = PolicyValueNet(board_width, board_height, ".\\CurrentModel\\GomokuAi")
    net_player = MCTSPlayer(brain.policy_value, 2000)
    mcts_player = MCTSPlayer(rollout_policy_value, 10000)

    while True:
        action = mcts_player.get_action(game_board)
        game_board.move(action)
        end, winner = game_board.check_winner()

        game_board.dbg_print()
        if end:
            print(winner)
            break

        action, prob = net_player.get_action_prob(game_board)
        game_board.move(action)
        end, winner = game_board.check_winner()
        print(action, prob)
        game_board.dbg_print()
        if end:
            print(winner)
            break


if __name__ == '__main__':
    net_mcts_play(11, 11, 5)