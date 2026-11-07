import torch
from torch.utils.data import random_split

class Group():
    """
    Base class for groups
    Adapted from Bilal Chugtai
    https://github.com/bilal-chughtai/rep-theory-mech-interp/blob/master/rep_theory/utils/groups.py
    """
    def __init__(self, index, order, identity,config):
        self.index = index                                  # Group index, i.e. 5 for S_5, 64 for Z_64
        self.order = order                                  # Number of elements in the group
        self.identity = identity                            # Identity element

        self.split = config.task.split
        self.seed = config.settings.seed

        self.multiplication_table = self.calculate_multiplication_table()
        self.inverses = self.compute_inverses()             # Where inverses[i] = inverse of element i

    def get_inverse(self, x):
        return (self.multiplication_table[x, :] == self.identity).nonzero().item()

    def calculate_multiplication_table(self):
        """
        Compute the operation result for every pair of elements.
        """
        table = torch.zeros((self.order, self.order), dtype=torch.int64)
        for i in range(self.order):
            for j in range(self.order):
                table[i, j] = self.compose(i,j)
        return table

    def compute_inverses(self):
        inverses = torch.zeros(self.order, dtype=torch.int64)
        for i in range(self.order):
            inverses[i] = self.get_inverse(i)
        return inverses



    def generate_train_test_data(self):
        """
        Generate training and testing datasets.
        """
        data = torch.zeros((self.order ** 2, 3), dtype=torch.int64)
        for i in range(self.order):
            for j in range(self.order):
                data[(i * self.order) + j] = torch.tensor([i, j, self.multiplication_table[i, j]])

        train_indices, test_indices = random_split(data, [self.split, 1-self.split],  generator=torch.Generator().manual_seed(self.seed))

        train = data[train_indices.indices]
        test = data[test_indices.indices]

        x_train = train[:, :2]
        y_train = train[:, 2]

        x_test = test[:, :2]
        y_test = test[:, 2]

        return x_train, y_train, x_test, y_test


