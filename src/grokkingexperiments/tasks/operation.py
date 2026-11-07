import torch
from torch.utils.data import random_split

class Operation():
  """
  Base to represent any mathematical binary operation
  """


  def __init__(self, config):
    self.num_elements = config.task.n
    self.split = config.task.split
    self.seed = config.settings.seed

    self.multiplication_table = self.calculate_multiplication_table()

  def compose(self, x, y):
    """
    Apply the binary operation to two elements.
    """
    raise NotImplementedError


  def calculate_multiplication_table(self):
    """
    Compute the operation result for every pair of elements.
    """
    table = torch.zeros((self.num_elements, self.num_elements), dtype=torch.int64)
    for i in range(self.num_elements):
        for j in range(self.num_elements):
            table[i, j] = self.compose(i,j)
    return table

  def generate_train_test_data(self):
    """
    Generate training and testing datasets.
    """
    data = torch.zeros((self.num_elements ** 2, 3), dtype=torch.int64)
    for i in range(self.num_elements):
        for j in range(self.num_elements):
            data[(i * self.num_elements) + j] = torch.tensor([i, j, self.multiplication_table[i, j]])

    train_indices, test_indices = random_split(data, [self.split, 1-self.split],  generator=torch.Generator().manual_seed(self.seed))

    train = data[train_indices.indices]
    test = data[test_indices.indices]

    x_train = train[:, :2]
    y_train = train[:, 2]

    x_test = test[:, :2]
    y_test = test[:, 2]

    return x_train, y_train, x_test, y_test
