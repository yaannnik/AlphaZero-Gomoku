"""
@author: yangyi
"""

from Gomoku.Chessboard import ChessBoard
from Gomoku.Gomoku import GomokuGame
from Player.ManualPlayer import ManualPlayer
from Player.AlphaZeroPlayer import AlphaZeroPlayer
from PytorchNet.PytorchNet import GomokuNet as PytorchNet
from KerasNet.KerasNet import GomokuNet as KerasNet
import pickle


def play_gomoku():
    size, n = 8, 5
    cb = ChessBoard(size, n)
    gomoku_game = GomokuGame(cb)
    pytorch_file = "./models/best_policy.pth"
    keras_file = "./models/best_policy.model"
    manual_player = ManualPlayer()

    pytorch_policy = PytorchNet(size, pytorch_file)
    pytorch_player = AlphaZeroPlayer(policy_value_func=pytorch_policy.board_policy_value,
                                     c_factor=5,
                                     n_playout=400)

    keras_policy = KerasNet(size, keras_file)
    keras_player = AlphaZeroPlayer(policy_value_func=keras_policy.board_policy_value,
                                   c_factor=5,
                                   n_playout=400)

    gomoku_game.start_play(pytorch_player, keras_player, start_player=0)


if __name__ == "__main__":
    play_gomoku()
