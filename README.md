# E6885-final-project (Gomoku Based on AlphaZero and Game Strategy Improvement)
Final Project of ELEN6885@Columbia University  
author: Jiashu Chen, Ziyu Liu, Lei Lyu, Chaoran Wei, Yi Yang, Zikai Zhu

## Descprition of project
Our project focus on the winning strategy of Gomoku Game by using Alpha Zero algorithm. We explore the influence of four factors on model performance. They are number of iterations, models, number of playouts, and UCB methods. Firstly, the number of iteration is proportional to the model performance. Secondly, the number of simulated games(playouts) also has a positive influence on the model performance. Additionally, complex models have a worse performance than simple model. Finally, different UCB method shows different convergence time and winning ratios.
## Organization of code  
```
.  
├── Gomoku  
│   ├── Chessboard.py  
│   ├── Gomoku.py   
│   └── __init__.py
├── KerasNet  
│   ├── KerasNet-4-layer-RNN.py  
│   ├── KerasNet.py  
│   ├── KerasNet18.py  
│   ├── KerasNet2.py  
│   └── __init__.py  
├── MonteCarlo  
│   ├── AlphaZero.py  
│   ├── MCTSNode.py  
│   ├── TreeNode.py  
│   ├── TreeSearch.py    
│   └── __init__.py
├── Player  
│   ├── AlphaZeroPlayer.py  
│   ├── MTCSPlayer.py  
│   ├── ManualPlayer.py   
│   └── __init__.py  
├── PytorchNet  
│   ├── PytorchNet.py   
│   └── __init__.py 
├── README.md  
├── play.py  
└── train.py   
```
## Description of code  
Gomoku: this file contains Gomoku experiment and a chess board to mantain chess aviliablity status and move.
KerasNet: this file contains four model built by Keras. They are original 3-layer model, 4-layer CNN model, 4-layer RNN model, and Resnet-18 model.  
MonteCarlo: this file is used to do MCTS search. We use two different UCB methods here.  
Player: the programs in this file are used to connect MCTS with CNN models. The CNN models take state as the input ang output policy and value for MCTS to do next move.  
PytorchNet: this file are models built by Pytorch.  
play.py: this file is used to do AI vs AI Gomoku Competition and explore winning strategy.
train.py: this file is used to train models.  