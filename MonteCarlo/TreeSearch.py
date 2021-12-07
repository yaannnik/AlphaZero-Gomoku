"""
@author: yangyi
"""

import numpy as np
import copy
from operator import itemgetter
from MonteCarlo.TreeNode import TreeNode


def rollout_policy(cb):
    """
    a coarse, fast version of policy_fn used in the rollout phase.
    parameters:
        cb - ChessBoard
    return:
        policy - zip(cb.vacants, act_probs)
    """
    # rollout randomly
    act_probs = np.random.rand(len(cb.vacants))
    return zip(cb.vacants, act_probs)


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


class MCTS(object):
    def __init__(self, policy_value_func, c_factor=5, n_playout=10000):
        """
        parameters:
            policy_value_fn - a function that takes in a board state and outputs
                a list of (action, probability) tuples and also a score in [-1, 1]
                (i.e. the expected value of the end game score from the current
                player's perspective) for the current player.
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

        act_probs, _ = self.policy(cb)
        # Check for end of game
        end, winner = cb.end_game()
        if not end:
            node.expand(act_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self.evaluate_rollout(cb)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def evaluate_rollout(self, cb, limit=1000):
        """
        Use the rollout policy to play until the end of the game,
        get award +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = cb.playing
        winner = 0
        for i in range(limit):
            end, winner = cb.end_game()
            if end:
                break
            act_probs = rollout_policy(cb)
            optimal_action = max(act_probs, key=itemgetter(1))[0]
            cb.move(optimal_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, cb):
        """
        Runs all playouts sequentially and returns the most visited action.
        parameters:
            state - the current game chessboard

        Return:
            action - the selected action
        """
        for n in range(self.n_playout):
            cb_copy = copy.deepcopy(cb)
            self.playout(cb_copy)
        return max(self.root.children.items(),
                   key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move):
        """
        Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_factor=5, n_playout=2000):
        self.mcts = MCTS(policy_value, c_factor, n_playout)
        self.player = 0

    def set_id(self, p):
        self.player = p

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
        return "MCTS {}".format(self.player)
