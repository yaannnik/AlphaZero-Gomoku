"""
@author: yangyi
"""

import random
import numpy as np
from collections import defaultdict, deque
from Gomoku.Chessboard import ChessBoard
from Gomoku.Gomoku import GomokuGame
from Player.MTCSPlayer import MCTSPlayer
from Player.AlphaZeroPlayer import AlphaZeroPlayer
from PytorchNet.PytorchNet import GomokuNet  # Pytorch
# from KerasNet.KerasNet import GomokuNet # Keras
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, weights=None):
        # params of the board and the game
        self.size = 5
        self.n = 3
        self.chess_board = ChessBoard(size=self.size, n=self.n)
        self.game = GomokuGame(self.chess_board)
        # training params
        self.lr = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_factor = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.iterations = 10
        self.best_win_percentage = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.mcts_playout = 1000
        self.episode_len = 0
        if weights:
            # start training from an initial policy-value net
            self.gomoku_net = GomokuNet(size=self.size,
                                        weights=weights)
        else:
            # start training from a new policy-value net
            self.gomoku_net = GomokuNet(size=self.size)

        self.alphazero_player = AlphaZeroPlayer(policy_value_func=self.gomoku_net.board_policy_value,
                                                c_factor=self.c_factor,
                                                n_playout=self.n_playout,
                                                is_self_play=1)

    def augment_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        augmented_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                new_state = np.array([np.rot90(s, i) for s in state])
                new_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.size, self.size)), i)
                augmented_data.append((new_state,
                                       np.flipud(new_mcts_prob).flatten(),
                                       winner))
                # flip horizontally
                new_state = np.array([np.fliplr(s) for s in new_state])
                new_mcts_prob = np.fliplr(new_mcts_prob)
                augmented_data.append((new_state,
                                       np.flipud(new_mcts_prob).flatten(),
                                       winner))
        return augmented_data

    def collect_self_play_data(self, n_games=1):
        """
        collect self-play data for training
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(player=self.alphazero_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.augment_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """
        update the policy-value net
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch, mcts_probs_batch, winner_batch = \
            [data[0] for data in mini_batch], [data[1] for data in mini_batch], [data[2] for data in mini_batch]
        old_probs, old_v = self.gomoku_net.batch_policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.gomoku_net.train_step(
                states=state_batch,
                mc_probs=mcts_probs_batch,
                winners=winner_batch,
                lr=self.lr * self.lr_multiplier)
            new_probs, new_v = self.gomoku_net.batch_policy_value(state_batch)
            kl = np.mean(np.sum(old_probs *
                                (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},\n"
               "lr_multiplier:{:.3f},\n"
               "loss:{},\n"
               "entropy:{},\n"
               "explained_var_old:{:.3f},\n"
               "explained_var_new:{:.3f}\n"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        alphazero_player = AlphaZeroPlayer(policy_value_func=self.gomoku_net.board_policy_value,
                                           c_factor=self.c_factor,
                                           n_playout=self.n_playout)
        mcts_player = MCTSPlayer(c_factor=5,
                                 n_playout=self.mcts_playout)
        wins = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(player1=alphazero_player,
                                          player2=mcts_player,
                                          start_player=i % 2,
                                          visualize=0)
            wins[winner] += 1
        win_percentage = 1.0 * (wins[1] + 0.5 * wins[-1]) / n_games
        print("num_playout:{}, win: {}, lose: {}, tie:{}".format(self.mcts_playout,
                                                                 wins[1], wins[2], wins[-1]))
        return win_percentage

    def run(self):
        """
        run the training pipeline
        """
        try:
            loss_, entropy_, iters_ = [], [], []
            for i in range(self.iterations):
                iter = i + 1
                self.collect_self_play_data(self.play_batch_size)
                print("batch:%d, episode_length:%d" % (iter, self.episode_len))

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    loss_.append(loss)
                    entropy_.append(entropy)
                    iters_.append(iter)

                # check the performance of the current model,
                # and save the model params
                if (i+1) % 100 == 0:
                    print("current self-play batch: %d" % iter)
                    win_percentage = self.policy_evaluate()
                    self.gomoku_net.save_model('./current_policy.pth')

                    if win_percentage > self.best_win_percentage:
                        print("New best policy!")
                        self.best_win_percentage = win_percentage
                        # update the best_policy
                        self.gomoku_net.save_model('./best_policy.pth')

                        if (self.best_win_percentage == 1.0 and
                                self.mcts_playout < 5000):
                            self.mcts_playout += 1000
                            self.best_win_percentage = 0.0
            plt.plot(iters_, loss_, label="loss")
            plt.plot(iters_, entropy_, label="entropy")
            plt.legend(loc='upper right')
            plt.xlabel("iterations")
            plt.title("Pytorch Net")
            # plt.show()
            plt.savefig("./PytorchNet.jpg")
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training = Trainer()
    training.run()