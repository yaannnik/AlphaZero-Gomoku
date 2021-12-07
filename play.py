"""
@author: yangyi
"""

from Gomoku.Chessboard import ChessBoard
from Gomoku.Gomoku import GomokuGame
from Player.ManualPlayer import ManualPlayer
from Player.AlphaZeroPlayer import AlphaZeroPlayer
from PytorchNet.PytorchNet import GomokuNet as PytorchNet


def play_gomoku():
    size, n = 8, 5
    cb = ChessBoard(8, 5)
    gomoku_game = GomokuGame(cb)
    pytroch_file = "./best_policy.pth"
    policy = PytorchNet(size, pytroch_file)
    alphazero_player1 = AlphaZeroPlayer(policy_value_func=policy.board_policy_value,
                                        c_factor=5,
                                        n_playout=400)
    alphazero_player2 = AlphaZeroPlayer(policy_value_func=policy.board_policy_value,
                                        c_factor=5,
                                        n_playout=400)
    # manual_player = ManualPlayer()

    gomoku_game.start_play(alphazero_player1, alphazero_player2, start_player=1)


if __name__ == "__main__":
    play_gomoku()