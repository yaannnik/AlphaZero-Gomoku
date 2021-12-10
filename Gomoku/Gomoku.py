"""
@author: yangyi
"""

import numpy as np


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
                    print('.'.center(8), end='')
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
        """
        start a self-play game using a MCTS player, reuse the search tree,
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
