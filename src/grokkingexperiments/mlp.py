import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):
    """
    Single hidden layer Multi Layer Perceptron
    Adapted from Bilal Chugtai
    https://github.com/bilal-chughtai/rep-theory-mech-interp/blob/master/rep_theory/utils/models.py
    """

    def __init__(self, config):
        super().__init__()

        torch.manual_seed(config.settings.seed)

        n = config.task.n
        embed_dim = config.model.embed_dim
        hidden_dim = config.model.hidden_dim

        self.left_embed = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.right_embed = nn.Parameter(torch.randn(n, embed_dim)/np.sqrt(embed_dim))
        self.hidden = nn.Parameter(torch.randn(2*embed_dim, hidden_dim)/np.sqrt(2*embed_dim))
        self.relu = nn.ReLU()
        self.unembedding = nn.Parameter(torch.randn(hidden_dim, n)/np.sqrt(hidden_dim))


    def forward(self, input):
        a_embed = self.left_embed[input[:, 0]]
        b_embed = self.right_embed[input[:, 1]]

        x = torch.cat((a_embed,b_embed), dim=1)
        x = x @ self.hidden
        x = self.relu(x)
        logits = x @ self.unembedding
        return logits
