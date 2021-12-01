#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Chaoran Wei
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy



class SelfPlay(object):
    """
    self play
    """

    def __init__(self):
        self.player = None

    def get_action(self, board):
        location = input("Your move: ")
        if isinstance(location, str):  # for python3
            location = [int(n, 10) for n in location.split(",")]
        move = board.location_to_move(location)
        if move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move


def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'

    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    # AI VS AI
    policy_param = pickle.load(open(model_file, 'rb'),encoding='bytes')
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,c_puct=5,n_playout=400)

    # set start_player
    game.start_self_play(mcts_player, is_shown=1)


run()