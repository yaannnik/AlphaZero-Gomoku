"""
@author: yangyi
"""

import sys
import os
import numpy as np


class ChessBoard:
    def __init__(self, size, n):
        self.size = size
        self.N = n
        # 1 for player 0 and 2 for player 1
        self.vacants = [i for i in range(self.size ** 2)]

        self.players = [1, 2]
        self.playing = self.players[0]  # 1 or 2

        self.pre_move = -1
        self.history = {}

    def reset(self, player=0):
        self.playing = self.players[player]
        self.vacants = [i for i in range(self.size ** 2)]
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
        if idx not in self.vacants:
            raise BaseException('Invalid move')

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
        win, winner = self.player_win()

        if win:
            return True, winner
        # Tie
        elif len(self.vacants) == 0:
            return True, -1

        return False, -1


class GomokuGame(object):
    def __init__(self, cb):
        self.chess_board = cb

    def show(self, p1, p2):
        print("Player %d with O" % p1)
        print("Player %d with X" % p2)
        print()
        for x in range(self.chess_board.size):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(self.chess_board.size - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(self.chess_board.size):
                loc = i * self.chess_board.size + j
                p = self.chess_board.history.get(loc, -1)
                if p == p1:
                    print('O'.center(8), end='')
                elif p == p2:
                    print('X'.center(8), end='')
                else:
                    print('+'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, visualize=True):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.chess_board.reset(start_player)

        ps = self.chess_board.players
        player1.set_id(ps[0])
        player2.set_id(ps[1])
        players = {ps[0]: player1, ps[1]: player2}
        if visualize:
            self.show(player1.id, player2.id)
        while True:
            player = players[self.chess_board.playing]
            move = player.get_action(self.chess_board)
            self.chess_board.move(move)

            if visualize:
                self.show(player1.id, player2.id)
            end, winner = self.chess_board.end_game()

            if end:
                if visualize:
                    if winner != -1:
                        print("Game end. Winner is: ", players[winner])
                    else:
                        print("Game end. No one wins")
                return winner

    def start_self_play(self, player, visualize=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.chess_board.reset()
        p1, p2 = self.chess_board.players
        states, mcts_probs, step_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.chess_board,
                                                 temp=temp,
                                                 return_prob=1)

            # record game history
            states.append(self.chess_board.get_state())
            mcts_probs.append(move_probs)
            step_players.append(self.chess_board.playing)
            # perform a move
            self.chess_board.move(move)

            if visualize:
                self.show(p1, p2)
            end, winner = self.chess_board.end_game()

            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(step_players))

                if winner != -1:
                    winners_z[np.array(step_players) == winner] = 1.0
                    winners_z[np.array(step_players) != winner] = -1.0
                # reset Monte-Carlo Tree Search root node
                player.reset_player()

                if visualize:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
