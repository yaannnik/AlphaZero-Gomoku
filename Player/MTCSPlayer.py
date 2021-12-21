"""
@author: yangyi
"""

import numpy as np
from MonteCarlo.TreeNode import TreeNode
from MonteCarlo.TreeSearch import MCTS


def policy_value(cb):
    """
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    parameters:
        cb - ChessBoard
    return:
        policy, value - zip(cb.vacants, act_probs), 0
    """
    # return uniform probabilities and 0 score for pure MCTS
    act_probs = np.ones(len(cb.vacants)) / len(cb.vacants)
    return zip(cb.vacants, act_probs), 0


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, C=5, n_playout=2000):
        self.mcts = MCTS(policy_value, C, n_playout)
        self.id = 0

    def set_id(self, p):
        self.id = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, cb):
        vacant_moves = cb.vacants
        if len(vacant_moves) > 0:
            move = self.mcts.get_move(cb)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: No Vacant Position")

    def __str__(self):
        return "MCTS player %d" % self.id
