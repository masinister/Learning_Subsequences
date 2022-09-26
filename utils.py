import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)

def bounded_diff_indices(n, k, g):
    assert n > k * (g + 1)
    indices = [0]
    while len(indices) < k:
        indices.append(indices[-1] + random.randint(1, g + 1))
    return np.array(indices) + random.randint(0, n - indices[-1] - 1)

def random_bin_sequence(n, planted = None, g = 0):
    seq = np.random.choice([0, 1], size=(n,))
    if planted is not None:
        mask = bounded_diff_indices(n, len(planted), g)
        seq[mask] = planted
    return seq

def planted_set(n, m, g, seq):
    S = []
    while len(S) < m:
        s = random_bin_sequence(n, seq, g)
        S.append(s)
    return np.vstack(S)

def non_planted_set(n, m, seq):
    S = []
    while len(S) < m:
        s = random_bin_sequence(n)
        if not is_subseq(s, seq):
            S.append(s)
    return np.vstack(S)

def train_test_split(dataset, test_split, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    test_idx = np.random.choice(indices, size=test_size, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
    testloader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler)

    return trainloader, testloader

if __name__ == '__main__':
    planted = random_bin_sequence(5)
    # print(random_bin_sequence(32, planted, 2))
    print(planted_set(16, 5, 2, planted))
    print(non_planted_set(16, 5, planted))
