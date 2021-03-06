"""
@author: yangyi
"""

import numpy as np


class TreeNode(object):
    def __init__(self, parent, prob):
        self.parent = parent  # prev TreeNode
        self.children = {}  # {action: next_node}
        self.num_visits = 0
        self.Q = 0
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

    def select(self, C):
        """
        Select action among children with optimal action value Q + ucb
        Return:
            A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda children: children[1].cal_ucb(C))

    def update(self, return_value):
        """
        Update node values from leaf evaluation.

        parameters:
            leaf_value - the value of subtree evaluation from the current player's perspective.
        """
        # Count visit.
        self.num_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (return_value - self.Q) / self.num_visits

    def update_recursive(self, return_value):
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-return_value)
        self.update(return_value)

    def cal_ucb(self, C):
        """
        Calculate and return the value for this node. It is a combination of leaf evaluations Q, and this node's
        upper confidence bound.

        parameters:
            C - a number in (0, inf) controlling the relative
                     impact of value Q, and prior probability P, on this node's score.
        """
        return self.Q + (C * self.P * np.sqrt(self.parent.num_visits) / (1 + self.num_visits))
        # (C * self.P * np.sqrt(np.log(self.parent.num_visits)) / (1 + self.num_visits))

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
