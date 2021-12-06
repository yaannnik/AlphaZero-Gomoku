from math import inf
import numpy as np
import copy


def estimate_update(old, new, stepsize):
    return old + stepsize * (new - old)


class MCTSNode:
    def __init__(self, parent, n):
        self.parent = parent
        self.children = {}
        self.total_n = n
        self.node_n = 0
        self.reward = 0

    def is_root(self):
        if self.parent:
            return False
        else:
            return True

    def is_leaf(self):
        if self.children:
            return False
        else:
            return True

    def expansion(self, nextaction):
        self.children[nextaction] = MCTSNode(self, self.total_n)

    def get_UCBvalue(self, C):
        if self.node_n == 0:
            return inf
        else:
            return self.reward + C * np.sqrt(np.log(self.total_n) / self.node_n)

    def children_selection(self, C):
        max_set = []
        max_value = 0
        for action in self.children:
            self.children[action].total_n = self.total_n
            if (self.children[action].get_UCBvalue(C) > max_value):
                max_set.clear()
                max_set.append(action)
                max_value = self.children[action].get_UCBvalue(C)
            elif (self.children[action].get_UCBvalue(C) == max_value):
                max_set.append(action)
        next_action = np.random.choice(len(max_set))
        return max_set[next_action], self.children[max_set[next_action]]

    def back_update(self, reward):
        self.total_n += 1
        self.node_n += 1
        self.reward = estimate_update(self.reward, reward, 1 / self.node_n)

    def back_propagation(self, reward):
        if not self.is_root():
            self.parent.back_propagation(-reward)
        self.back_update(reward)


class MCTS:
    def __init__(self, state, numplay, C=1):
        self.root = MCTSNode(None, 0)
        self.state = state
        self.numplay = numplay
        self.C = C

    def one_play(self, state_sim):
        current_node = self.root
        while not current_node.is_leaf():
            next_action, current_node = current_node.children_selection(self.C)
            state_sim.move(next_action)
        End, _ = state_sim.endGame()
        if not End:
            for vacant in state_sim.vacants:
                current_node.expansion(vacant)
            rand_choice = np.random.choice(len(state_sim.vacants))
            next_action = state_sim.vacants[rand_choice]
            current_node = current_node.children[next_action]
            state_sim.move(next_action)
        simulation_winner = self.simulation(state_sim)
        current_node.back_propagation(simulation_winner)

    def simulation(self, state_sim):
        End, Winner = state_sim.end_game()
        new_leaf_player = state_sim.playing
        while not End:
            rand_choice = np.random.choice(len(state_sim.vacants))
            next_action = state_sim.vacants[rand_choice]
            state_sim.move(next_action)
            End, Winner = state_sim.end_game()
        if Winner == -1:
            return 0
        elif Winner == new_leaf_player:
            return 1
        else:
            return -1

    def move_from_root(self):
        for play in range(self.numplay):
            state_sim = copy.deepcopy(self.state)
            self.one_play(state_sim)
        next_actions = []
        max_visit = 0
        for action in self.root.children:
            if (self.root.children[action].node_n > max_visit):
                next_actions.clear()
                next_actions.append(action)
                max_visit = self.root.children[action].node_n
            elif (self.root.children[action].node_n == max_visit):
                next_actions.append(action)
        next_action = np.random.choice(len(next_actions))
        return next_actions[next_action]


class MCTSPlayer:
    def __init__(self, state, numplay=1000, C=1):
        self.state = state
        self.numplay = numplay
        self.mcts = MCTS(state, numplay, C)

    def reset(self):
        self.mcts.root = MCTSNode(None, 0)

    def get_move(self):
        if self.state.vacants:
            move = self.mcts.move_from_root()
            self.reset()
            return move

        return -1
