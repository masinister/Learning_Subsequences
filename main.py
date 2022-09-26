import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os

from models import FC_NN, Conv_NN
from dataset import PlantedSubsequenceDataset
from learning import train, test
from utils import train_test_split, random_bin_sequence

device = torch.device("cuda")
n = 10
k = int(n/2)
g = 0
seq = random_bin_sequence(k)

batch_size = 128
num_epochs = 10

model = FC_NN(n).to(device)

dataset = PlantedSubsequenceDataset(n = n, g = g, seq = seq, device = device)

trainloader, testloader = train_test_split(dataset, 0.9, batch_size)

print(len(dataset), len(trainloader), len(testloader), batch_size)

model = train(model, trainloader, testloader, device, num_epochs)

torch.save(model.state_dict(),"testmodel.pth")
