'''
@author: yangyi
'''
import sys
import os
import numpy as np

class ChessBoard:
    def __init__(self, size):
        self.size = size
        # '0' for player 0 and '1' for player 1
        self.pieces = ['0', '1']
        self.chess_board = [['#' for _ in range(size)] for _ in range(size)]
        self.vacants = [i for i in range(size**2)]

        self.player = -1
        self.pre_move = -1
        self.records = {}

    # def state(self):
    #     square_state = np.zeros((4, self.size, self.size))

    #     if self.records:
    #         moves, players = np.array(list(zip(*self.records.items())))

    #         move_self = moves[players == self.player]
    #         move_oppo = moves[players != self.player]

    #         square_state[0][move_self // self.size,
    #                         move_self % self.size] = 1.0
    #         square_state[1][move_oppo // self.size,
    #                         move_oppo % self.size] = 1.0
    #         # indicate the last move location
    #         square_state[2][self.pre_move // self.size,
    #                         self.pre_move % self.size] = 1.0

    #     if len(self.records) % 2 == 0:
    #         square_state[3][:, :] = 1.0  # indicate the player to play

    #     return square_state[:, ::-1, :]

    def move(self, player, row, col):
        if self.player == player:
            raise BaseException('Change player')

        self.player = player
        piece = self.pieces[player % 2]

        if self.chess_board[row % self.size][col % self.size] != '#':
            raise BaseException('Occupied move')

        self.chess_board[row % self.size][col % self.size] = piece
        self.records[row * self.size + col] = self.player
        self.pre_move = row * self.size + col
        self.vacants.remove(row * self.size + col)

    def playerWin(self):
        records = self.records
        n = 5

        moves = list(set(records.keys()))
        print(moves)
        if len(moves) < n * 2 - 1:
            return -1

        for m in moves:
            row = m // self.size
            col = m % self.size

            player = records[m]

            if (col in range(self.size - n + 1) and
                    len(set(records.get(i, -1) for i in range(m, m + n))) == 1):
                return player

            if (row in range(self.size - n + 1) and
                    len(set(records.get(i, -1) for i in range(m, m + n * self.size, self.size))) == 1):
                return player

            if (col in range(self.size - n + 1) and row in range(self.size - n + 1) and
                    len(set(records.get(i, -1) for i in range(m, m + n * (self.size + 1), self.size + 1))) == 1):
                return player

            if (col in range(n - 1, self.size) and row in range(self.size - n + 1) and
                    len(set(records.get(i, -1) for i in range(m, m + n * (self.size - 1), self.size - 1))) == 1):
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
