"""

by Hao Xue @ 10/04/18

"""

import numpy as np
from indrnn import IndRNN
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data


# prepare data
data = np.load('/home/ubuntu/Desktop/indrnn-pytorch/data/data_40frames.npy')
train_data = data[0: 4000]
test_data = data[4000: 4750]
train_input = train_data[:, 0: 20]
train_label = train_data[:, 20: 40]
test_input = test_data[:, 0: 20]
test_label = test_data[:, 20: 40]

dataset = Data.TensorDataset(data_tensor=Variable(torch.Tensor(train_input)).cuda(),
                             target_tensor=Variable(torch.Tensor(train_label)).cuda())
dataloader = Data.DataLoader(
    dataset=dataset,
    batch_size=50,
    shuffle=True,
    num_workers=1,
)
# hyper parameters
TIME_STEPS = 20
PREDICT_STEPS = 20
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
epoch = 3
cuda = torch.cuda.is_available()


# build predictor
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = IndRNN(input_size=input_size,
                              hidden_size=hidden_size,
                              n_layer=num_layers,
                              batch_norm=False,
                              hidden_max_abs=RECURRENT_MAX,
                              step_size=TIME_STEPS)

        self.decoder = IndRNN(input_size=hidden_size,
                              hidden_size=hidden_size,
                              n_layer=num_layers,
                              batch_norm=False,
                              hidden_max_abs=pow(2, 1 / PREDICT_STEPS),
                              step_size=PREDICT_STEPS)

        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden=None):
        y = self.encoder(x, hidden)

        temp = y[:, -1].expand(PREDICT_STEPS, -1, self.hidden_size)

        out = self.decoder(temp, hidden)

        predict = self.output(out[:, 0])

        return predict


def main():
    # build model

    model = Net()

    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    loss_func = nn.MSELoss()
    while step < epoch:
        losses = []

        for i, (batch_x, batch_y) in enumerate(dataloader):
            data, target = batch_x, batch_y
            model.zero_grad()
            out = model(data)
            loss = loss_func(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu()[0])
        step += 1
        if step >= epoch:
            break
        if epochs % 1 == 0:
            print(
                "Epoch {} cross_entropy {}".format(
                    epochs, np.mean(losses)))
        epochs += 1


main()
torch.multiprocessing.set_start_method("spawn")
