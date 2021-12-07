"""
@author: yangyi
"""

import numpy as np
import copy
from MonteCarlo.TreeNode import TreeNode


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS(object):
    def __init__(self, policy_value_func, c_factor=5, n_playout=10000):
        """
        parameters:
            policy_value_func - get policy (action and prob) and value
            c_factor - a number in (0, inf) that controls how quickly exploration
                converges to the maximum-value policy. A higher value means
                relying on the prior more.
        """
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_func
        self.c_factor = c_factor
        self.n_playout = n_playout

    def playout(self, cb):
        """
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self.c_factor)
            cb.move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        act_probs, leaf_value = self.policy(cb)
        # Check for end of game.
        end, winner = cb.end_game()
        if not end:
            node.expand(act_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == cb.playing else -1.0)

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, cb, temp=1e-3):
        """
        Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        parameters:
            state - the current game state
            temp - temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self.n_playout):
            cb_copy = copy.deepcopy(cb)
            self.playout(cb_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, prev_move):
        """
        Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if prev_move in self.root.children:
            self.root = self.root.children[prev_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


