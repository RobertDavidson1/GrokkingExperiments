
import torch
from ..task import Task


class Group(Task):
    """
    Base class for groups
    Adapted from Bilal Chugtai
    https://github.com/bilal-chughtai/rep-theory-mech-interp/blob/master/rep_theory/utils/groups.py
    """
    def __init__(self, index, order, identity,config):
        self.index = index                                  # Group index, i.e. 5 for S_5, 64 for Z_64
        self.order = order                                  # Number of elements in the group
        self.identity = identity                            # Identity element

        super().__init__(config)
        self.inverses = self.compute_inverses()             # Where inverses[i] = inverse of element i

    def get_inverse(self, x):
        return (self.multiplication_table[x, :] == self.identity).nonzero().item()

    def compute_inverses(self):
        inverses = torch.zeros(self.order, dtype=torch.int64)
        for i in range(self.order):
            inverses[i] = self.get_inverse(i)
        return inverses

