"""
@author: yangyi
"""

from Gomoku.Chessboard import ChessBoard
from Gomoku.Gomoku import GomokuGame
from Player.ManualPlayer import ManualPlayer
from Player.AlphaZeroPlayer import AlphaZeroPlayer
from PytorchNet.PytorchNet import GomokuNet as PytorchNet
from KerasNet.KerasNet import GomokuNet as KerasNet
from KerasNet.KerasNet2 import GomokuNet2 as KerasNet2
from KerasNet.KerasNet18 import GomokuNet18 as KerasNet18
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def play_gomoku():
    size, n = 8, 5
    cb = ChessBoard(size, n)
    gomoku_game = GomokuGame(cb)
    pytorch_file = "./models/best_policy.pth"
    keras_file = "./models/best_policy.model"
    keras_file2 = "models/Keras_100_Resnet2.model"
    keras_file18 = "models/Keras_100_Resnet18.model"
    manual_player = ManualPlayer()

    pytorch_policy = PytorchNet(size, pytorch_file)
    pytorch_player = AlphaZeroPlayer(policy_value_func=pytorch_policy.board_policy_value,
                                     C=5,
                                     n_playout=400)

    keras_policy = KerasNet(size, keras_file)
    keras_player = AlphaZeroPlayer(policy_value_func=keras_policy.board_policy_value,
                                   C=5,
                                   n_playout=400)

    keras_policy2 = KerasNet2(size, keras_file2)
    keras_player2 = AlphaZeroPlayer(policy_value_func=keras_policy2.board_policy_value,
                                    C=5,
                                    n_playout=400)

    keras_policy18 = KerasNet18(size, keras_file18)
    keras_player18 = AlphaZeroPlayer(policy_value_func=keras_policy18.board_policy_value,
                                     C=5,
                                     n_playout=100)

    gomoku_game.start_play(pytorch_player, keras_player, first_move=1)


if __name__ == "__main__":
    play_gomoku()
