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
m = 2048
n = 256
k = int(n / 16)
# g = 4

batch_size = 8
num_epochs = 16

fc_model = FC_NN(n).to(device)
conv_model = Conv_NN(n, k).to(device)
torch.save(fc_model.state_dict(), 'FC_init.pt')
torch.save(conv_model.state_dict(), 'Conv_init.pt')

fc_accs = []
conv_accs = []

for g in range(8):
    print("##########################################")
    print("############ g = {}          #############".format(g))
    print("##########################################")

    fc_model.load_state_dict(torch.load('FC_init.pt'))
    conv_model.load_state_dict(torch.load('Conv_init.pt'))

    dataset = PlantedSubsequenceDataset(m, n, k, g, device = device)

    trainloader, testloader = train_test_split(dataset, 0.5, batch_size)

    print(len(dataset), len(trainloader), len(testloader), batch_size)

    model, conv_acc = train(conv_model, trainloader, testloader, device, num_epochs)
    model, fc_acc = train(fc_model, trainloader, testloader, device, num_epochs)

    fc_accs.append(fc_acc[-1])
    conv_accs.append(conv_acc[-1])

    # plt.figure(0)
    # plt.title("Learning curve")
    # plt.xlabel("Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(conv_acc,  label='Conv_NN')
    # plt.plot(fc_acc,  label='FC_NN')
    # plt.legend()
    # plt.show()
plt.figure(0)
plt.title("Accuracy")
plt.xlabel("Maximum planted subsequence gap")
plt.ylabel("Validation Accuracy")
plt.plot(conv_accs,  label='Conv_NN')
plt.plot(fc_accs,  label='FC_NN')
plt.legend()
plt.show()
