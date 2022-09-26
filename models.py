import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class FC_NN(nn.Module):

    def __init__(self, n):
        super(FC_NN, self).__init__()
        self.n = n

        self.fc = nn.Sequential(nn.Linear(self.n, self.n),
                                nn.ReLU(),
                                nn.Linear(self.n, self.n),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(self.n, 2))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

class Conv_NN(nn.Module):

    def __init__(self, n):
        super(Conv_NN, self).__init__()
        self.n = n

        self.fc = nn.Sequential(nn.Linear(self.n, self.n),
                                nn.ReLU(),
                                nn.Linear(self.n, self.n),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(self.n, 2))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
