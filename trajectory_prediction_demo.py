"""

by Hao Xue @ 10/04/18

"""

import numpy as np
from indrnn import IndRNN
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# prepare data
data = np.load('/home/ubuntu/Desktop/indrnn-pytorch/data/data_40frames.npy')
train_data = data[0: 4000]
test_data = data[4000: 4750]
train_input = train_data[:, 0: 20]
train_label = train_data[:, 20: 40]
test_input = test_data[:, 0: 20]
test_label = test_data[:, 20: 40]

print(data[:, -1])

# hyper parameters
# TIME_STEPS = 20
# RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
# cuda = torch.cuda.is_available()
#
#
# # build predictor
# class Net(nn.Module):
#     def __init__(self, input_size=2, hidden_size=128, num_layers=2):
#         super(Net, self).__init__()
#         self.indrnn = IndRNN(input_size=input_size,
#                              hidden_size=hidden_size,
#                              n_layer=num_layers,
#                              batch_norm=False,
#                              hidden_max_abs=RECURRENT_MAX,
#                              step_size=TIME_STEPS)
#
#     def forward(self, x, hidden=None):
#
#         y = self.indrnn(x, hidden)
#
