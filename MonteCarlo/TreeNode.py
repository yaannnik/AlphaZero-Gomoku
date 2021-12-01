"""
@author: yangyi
"""

import numpy as np
import copy
from operator import itemgetter


class TreeNode(object):
    def __init__(self, parent, prob):
        self.parent = parent  # prev TreeNode
        self.children = {}  # {action: next_node}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prob

    def expand(self, action_prob):
        """
        Parameters:
            action_priors - a list of tuples of actions and their prior probability
                            according to the policy function.
        """
        for action, prob in action_prob:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, factor):
        """
        Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return:
            A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda children: children[1].get_value(factor))

    def update(self, return_value):
        """
        Update node values from leaf evaluation.

        parameters:
            leaf_value - the value of subtree evaluation from the current player's perspective.
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (return_value - self.Q) / self.n_visits

    def update_recursive(self, return_value):
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-return_value)
        self.update(return_value)

    def get_value(self, factor):
        """
        Calculate and return the value for this node. It is a combination of leaf evaluations Q, and this node's
        prior adjusted for its visit count, u.

        parameters:
            factor - a number in (0, inf) controlling the relative
                     impact of value Q, and prior probability P, on this node's score.
        """
        self.u = (factor * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
