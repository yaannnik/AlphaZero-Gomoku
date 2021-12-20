"""
@author: yangyi
"""

import numpy as np


class ChessBoard:
    def __init__(self, size, n):
        self.size = size
        self.N = n
        # 1 for player 0 and 2 for player 1
        self.vacants = [i for i in range(self.size * self.size)]

        self.players = [1, 2]
        self.playing = self.players[0]  # 1 or 2

        self.pre_move = -1
        self.history = {}

    def reset(self, player=0):
        self.playing = self.players[player]
        self.vacants = [i for i in range(self.size * self.size)]
        self.history = {}
        self.pre_move = -1

    def get_state(self):
        state = np.zeros((4, self.size, self.size))

        if self.history:
            moves, players = np.array(list(zip(*self.history.items())))

            move_self = moves[players == self.playing]
            move_oppo = moves[players != self.playing]

            state[0][move_self // self.size,
                     move_self % self.size] = 1.0
            state[1][move_oppo // self.size,
                     move_oppo % self.size] = 1.0
            # indicate the last move location
            state[2][self.pre_move // self.size,
                     self.pre_move % self.size] = 1.0

        if len(self.history) % 2 == 0:
            state[3][:, :] = 1.0  # indicate the player to play

        return state[:, ::-1, :]

    def move(self, idx):
        self.history[idx] = self.playing
        self.pre_move = idx
        self.vacants.remove(idx)

        self.playing = self.playing % 2 + 1  # change player

    def player_win(self):
        history = self.history
        n = self.N

        moves = list(set(history.keys()))

        if len(moves) < n * 2 - 1:
            return False, -1

        for m in moves:
            row = m // self.size
            col = m % self.size

            player = history[m]

            if (col in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * self.size, self.size))) == 1):
                return True, player

            if (col in range(self.size - n + 1) and row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * (self.size + 1), self.size + 1))) == 1):
                return True, player

            if (col in range(n - 1, self.size) and row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * (self.size - 1), self.size - 1))) == 1):
                return True, player

        return False, -1

    def end_game(self):
        """
        return:
            end, winner - True is any player wins or tie, -1 if tie
        """
        win, winner = self.player_win()

        if win:
            return True, winner
        # Tie
        elif len(self.vacants) == 0:
            return True, -1

        return False, -1