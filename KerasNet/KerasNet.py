#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Chaoran Wei, Ziyu Liu
"""

from __future__ import print_function
from keras import Input, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import adam_v2
import keras.backend as K
import numpy as np
import pickle


class GomokuNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # l2 penalty
        self.create_policy_value_net()
        self._loss_train_op()

    def create_policy_value_net(self):
        """create the policy value network by Keras"""
        inpu = Input((4, self.board_width, self.board_height))

        # conv layers
        Conv = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_last", activation="relu",
                      kernel_regularizer=l2(self.l2_const))(inpu)
        Conv = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_last", activation="relu",
                      kernel_regularizer=l2(self.l2_const))(Conv)
        Conv = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_last", activation="relu",
                      kernel_regularizer=l2(self.l2_const))(Conv)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                            kernel_regularizer=l2(self.l2_const))(Conv)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width * self.board_height, activation="softmax",
                                kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                           kernel_regularizer=l2(self.l2_const))(Conv)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(inpu, [self.policy_net, self.value_net])

        def sample_policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results

        self.sample_policy_value = sample_policy_value

    def board_policy_value(self, cb):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = cb.availables
        current_state = cb.get_state()
        act_probs, value = self.sample_policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = adam_v2.Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(s, pi, z, lr):
            state_input_union = np.array(s)
            mcts_probs_union = np.array(pi)
            winner_union = np.array(z)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union],
                                       batch_size=len(s), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, lr)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(s), verbose=0)
            return loss[0], entropy

        self.train_step = train_step

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.model.get_weights()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
