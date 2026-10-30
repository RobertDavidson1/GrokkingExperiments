from torch.utils.data import Dataset, random_split
import torch
from grokkingexperiments.groups import generate_data

class GrokkingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = torch.from_numpy(dataset).long()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx, :2], self.dataset[idx, 2]


def split_data(data, split):
    dataset = GrokkingDataset(data)
    train_indices, test_indices = random_split(dataset, [split, 1-split])

    x_train, y_train = dataset[train_indices.indices]
    x_test, y_test = dataset[test_indices.indices]

    return x_train, y_train, x_test, y_test

def generate_train_test_data(n, op, split):
    data = generate_data(n, op)
    return split_data(data, split)
