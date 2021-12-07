"""
@author: yangyi
"""

import numpy as np
from MonteCarlo.AlphaZero import MCTS


class AlphaZeroPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_func, c_factor=5, n_playout=2000, is_self_play=False):
        self.mcts = MCTS(policy_value_func, c_factor, n_playout)
        self.is_self_play = is_self_play
        self.id = 0

    def set_id(self, p):
        self.id = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, cb, temp=1e-3, return_prob=0):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(cb.size ** 2)

        if len(cb.vacants) > 0:
            acts, probs = self.mcts.get_move_probs(cb, temp)
            move_probs[list(acts)] = probs
            if self.is_self_play:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "AlphaZero player %d" % self.id
