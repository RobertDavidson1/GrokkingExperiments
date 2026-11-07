import torch
from torch.utils.data import random_split


class Task:
    def __init__(self, config):
        self.multiplication_table = self.calculate_multiplication_table()
        self.split = config.task.split

    def compose(self, x, y):
        raise NotImplementedError

    def calculate_multiplication_table(self):
            table = torch.zeros((self.order, self.order), dtype=torch.int64)
            for i in range(self.order):
                for j in range(self.order):
                    table[i, j] = self.compose(i,j)
            return table


    def generate_train_test_data(self):


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



