import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class FC_NN(nn.Module):

    def __init__(self, n):
        super(FC_NN, self).__init__()
        self.n = n

        self.fc = nn.Sequential(nn.Linear(self.n, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 2))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

class Conv_NN(nn.Module):

    def __init__(self, n, k):
        super(Conv_NN, self).__init__()
        self.n = n

        self.conv = nn.Sequential(nn.Conv1d(1, 4, int(k/2)),
                                  nn.ReLU(),
                                  )

        self.fc_size = np.prod(self.conv(torch.zeros((1,1,n))).shape[1:])

        self.fc = nn.Sequential(nn.Linear(self.fc_size, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 2)
                                )

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    net = Conv_NN(512,64)
    print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    net = FC_NN(512)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
