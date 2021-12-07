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
                leaf_value = (
                    1.0 if winner == cb.playing else -1.0
                )

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
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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


class AlphaZeroPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_func, c_factor=5, n_playout=2000, is_self_play=False):
        self.mcts = MCTS(policy_value_func, c_factor, n_playout)
        self.is_self_play = is_self_play
        self.player = 0

    def set_id(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, cb, temp=1e-3, return_prob=0):
        vacant_moves = cb.vacants
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(cb.size**2)

        if len(vacant_moves) > 0:
            acts, probs = self.mcts.get_move_probs(cb, temp)
            move_probs[list(acts)] = probs
            if self.is_self_play:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
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
        return "MCTS {}".format(self.player)