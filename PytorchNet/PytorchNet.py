"""
@author: yangyi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class PlainNet(nn.Module):
    def __init__(self, size):
        """
        parameters:
            size - chess board size
        """
        super(PlainNet, self).__init__()

        self.size = size

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear((self.size ** 2) * 4, self.size ** 2)

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear((self.size ** 2) * 2, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, input):
        """
        parameters:
            input - (n, 4, size, size) input state matrix
        
        return:
            act_prob - probability matrix of (1, size*size) for actions
            state_val - matrix of (1, 1) for state values
        """

        # common layers
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, (self.size ** 2) * 4)
        act_prob = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, (self.size ** 2) * 2)
        x_val = F.relu(self.val_fc1(x_val))
        state_val = torch.tanh(self.val_fc2(x_val))

        return act_prob, state_val


class GomokuNet:
    def __init__(self, size, weights=None, device='cpu'):
        """
        parameters:
            size - chess board size
            weights - pre trained model weights
            device - cpu or cuda
        """
        self.device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        self.size = size
        self.weight_decay = 1e-4

        self.net = PlainNet(size).to(self.device)

        if weights is not None:
            self.net.load_state_dict(torch.load(weights))

        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.weight_decay)

    def sample_policy_value(self, states):
        """
        parameters:
            states - input states
        return:
            actor_probs - possibility map of moves(np.array)
            state_val - value map of states(np.array)
        """
        states = Variable(torch.FloatTensor(states).to(self.device))

        act_tensor, state_tensor = self.net(states)

        act_probs = np.exp(act_tensor.detach().cpu().numpy())
        state_val = state_tensor.detach().cpu().numpy()

        return act_probs, state_val

    def board_policy_value(self, cb):
        """
        parameters:
            cb - object of ChessBoard
        return:
            actor_probs - possibility map of moves(dict)
            state_val - value map of states(float)
        """
        state = np.ascontiguousarray(cb.get_state().reshape(-1, 4, self.size, self.size))

        act_scores, value = self.net(
            Variable(torch.FloatTensor(state)).to(self.device))
        act_probs = np.exp(act_scores.detach().cpu().numpy().flatten())

        act_probs = zip(cb.vacants, act_probs[cb.vacants])
        state_val = value.detach()[0][0]

        return act_probs, state_val

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_step(self, s, pi, z, lr):
        """
        train in each time step, (s, pi, z) is identical with AlphaZero paper
        parameters:
            s - state
            pi - policy
            z - winner
            lr - learning rate
        """
        s = Variable(torch.FloatTensor(s).to(self.device))
        pi = Variable(torch.FloatTensor(pi).to(self.device))
        z = Variable(torch.FloatTensor(z).to(self.device))

        self.optimizer.zero_grad()
        self.set_learning_rate(lr)

        # forward propagation
        p, v = self.net(s)
        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(v.view(-1), z)
        policy_loss = -torch.mean(torch.sum(pi * p, 1))
        loss = value_loss + policy_loss

        # backward propagation and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(p) * p, 1))

        return loss.item(), entropy.item()

    def save_model(self, model_path):
        """
        save model params to file 
        """
        params = self.net.state_dict()  # get model params
        torch.save(params, model_path)
