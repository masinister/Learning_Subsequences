import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from models import FC_NN, Conv_NN
from dataset import PlantedSubsequenceDataset
from learning import train, test
from utils import train_test_split, random_bin_sequence

device = torch.device("cuda")
m = 512
n = 512
k = int(n / 8)
g = 4

batch_size = 1
num_epochs = 16

fc_model = FC_NN(n).to(device)
conv_model = Conv_NN(n, k).to(device)

dataset = PlantedSubsequenceDataset(m, n, k, g, device = device)

trainloader, testloader = train_test_split(dataset, 0.5, batch_size)

print(len(dataset), len(trainloader), len(testloader), batch_size)

model, conv_acc = train(conv_model, trainloader, testloader, device, num_epochs)
model, fc_acc = train(fc_model, trainloader, testloader, device, num_epochs)

plt.figure(0)
plt.title("Learning curve")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(conv_acc,  label='Conv_NN')
plt.plot(fc_acc,  label='FC_NN')
plt.legend()
plt.show()
# torch.save(model.state_dict(),"testmodel.pth")
