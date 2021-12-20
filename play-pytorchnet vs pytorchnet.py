#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Chaoran Wei, Ziyu Liu
"""

from Gomoku.Chessboard import ChessBoard
from Gomoku.Gomoku import GomokuGame
from Player.ManualPlayer import ManualPlayer
from Player.AlphaZeroPlayer import AlphaZeroPlayer
from PytorchNet.PytorchNet import GomokuNet as PytorchNet
from KerasNet.KerasNet import GomokuNet as KerasNet
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def play_gomoku():
    size, n = 8, 5
    cb = ChessBoard(size, n)
    gomoku_game = GomokuGame(cb)
    pytorch_file = "models/PytorchNet-1500.pth"
    pytorch_file1 = "models/PytorchNet-1500.pth"

    pytorch_policy = PytorchNet(size, pytorch_file)
    pytorch_player = AlphaZeroPlayer(policy_value_func=pytorch_policy.board_policy_value,
                                     C=5,
                                     n_playout=50)

    pytorch_policy1 = PytorchNet(size, pytorch_file)
    pytorch_player1 = AlphaZeroPlayer(policy_value_func=pytorch_policy1.board_policy_value,
                                      C=5,
                                      n_playout=1000)

    gomoku_game.start_play(pytorch_player, pytorch_player1, start_player=1)


play_gomoku()
