import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)

def is_bounded_diff_seq(S, g):
    return all(abs(j - i) <= g + 1 for i,j in zip(S,S[1:]))

def contains_k_ones(x, k, g = 0):
    I = np.where(x == 1)[0]
    return any(is_bounded_diff_seq(S, g) for S in zip(*[I[i:] for i in range(k)]))

def random_bin_sequence(n, p1 = 0.5):
    return np.random.choice([0, 1], size=(n,), p = [1-p1, p1])

def random_planted_sequence(n, k, g = 0):
    seq = random_bin_sequence(n)
    while not contains_k_ones(seq, k, g):
        i = np.random.randint(n)
        for j in range(k):
            c = np.random.randint(g + 1)
            if i + j*(c+1) >= n:
                break
            seq[i + j*(c+1)] = 1
    return seq

def random_nonplanted_sequence(n, k, g = 0):
    seq = random_bin_sequence(n)
    while contains_k_ones(seq, k, g):
        I = np.where(seq == 1)[0]
        for i in range(len(I) - k):
            if contains_k_ones(seq[I[i]:I[i+k]+1], k, g):
                j = np.random.randint(k)
                seq[I[i+j]] = 0
    return seq

def planted_set(m, n, k, g):
    S = []
    print("Generating planted set")
    for i in tqdm(range(m)):
        s = random_planted_sequence(n, k, g)
        S.append(s)
    return np.vstack(S)

def non_planted_set(m, n, k, g):
    S = []
    print("Generating non-planted set")
    for i in tqdm(range(m)):
        s = random_nonplanted_sequence(n, k, g)
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
    print(np.mean(planted_set(50, 100, 5, 0)))
    print(np.mean(non_planted_set(50, 100, 5, 0)))
