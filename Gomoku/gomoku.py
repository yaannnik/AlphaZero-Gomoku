'''
@author: yangyi
'''

import sys
import os
import numpy as np

class ChessBoard:
    def __init__(self, size):
        self.size = size
        # 1 for player 1 and 2 for player 2
        self.pieces = ['x', 'o']
        self.chess_board = [['.' for _ in range(size)] for _ in range(size)]
        self.vacants = [i for i in range(size**2)]

        self.player = -1
        self.pre_move = -1
        self.history = {}


    def getState(self):
        state = np.zeros((4, self.size, self.size))

        if self.history:
            moves, players = np.array(list(zip(*self.history.items())))

            move_self = moves[players == self.player]
            move_oppo = moves[players != self.player]

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


    def move(self, row, col):
        self.player = (self.player + 1) % 2
        piece = self.pieces[self.player]

        idx = row * self.size + col

        if self.chess_board[row % self.size][col % self.size] != '#':
            raise BaseException('Occupied move')

        self.chess_board[row % self.size][col % self.size] = piece
        self.history[idx] = self.player
        self.pre_move = idx
        self.vacants.remove(idx)


    def playerWin(self):
        history = self.history
        n = 5

        moves = list(set(history.keys()))
        print(moves)
        if len(moves) < n * 2 - 1:
            return -1

        for m in moves:
            row = m // self.size
            col = m % self.size

            player = history[m]

            if (col in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n))) == 1):
                return player

            if (row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * self.size, self.size))) == 1):
                return player

            if (col in range(self.size - n + 1) and row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * (self.size + 1), self.size + 1))) == 1):
                return player

            if (col in range(n - 1, self.size) and row in range(self.size - n + 1) and
                    len(set(history.get(i, -1) for i in range(m, m + n * (self.size - 1), self.size - 1))) == 1):
                return player

        return -1


    def endGame(self):
        winner = self.playerWin()

        if winner != -1 or len(self.vacants) == 0:
            return True, winner

        return False, winner


    def show(self):
        for line in self.chess_board:
            print(line)
