import torch
from torch.utils.data import Dataset

from utils import planted_set, non_planted_set, random_bin_sequence

class PlantedSubsequenceDataset(Dataset):

    def __init__(self, n, g, seq, device = torch.device("cuda")):
        super(PlantedSubsequenceDataset, self).__init__()
        self.device = device
        self.n = n
        self.max_gap = g
        self.subsequence = seq
        self.planted = torch.tensor(planted_set(n, 10000, g, seq), dtype = torch.float, device = self.device)
        self.non_planted = torch.tensor(non_planted_set(n, 10000, seq), dtype = torch.float, device = self.device)

    def __getitem__(self, index):
        y = index % 2
        i = int(index/2)
        if y == 0:
            x = self.non_planted[i]
        else:
            x = self.planted[i]
        return x, torch.tensor(y, dtype = torch.long, device = self.device)

    def __len__(self):
        return len(self.planted) + len(self.non_planted)

if __name__ == '__main__':
    dataset = PlantedSubsequenceDataset(20, 0, random_bin_sequence(10))
    X, y = dataset[0]
    print(X, y)
    X, y = dataset[1]
    print(X, y)
