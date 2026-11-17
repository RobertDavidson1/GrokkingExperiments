import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Single hidden layer Multi Layer Perceptron
    Adapted from Bilal Chugtai
    https://github.com/bilal-chughtai/rep-theory-mech-interp/blob/master/rep_theory/utils/models.py
    """
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.settings.seed)

        n = config.group.n                      # |G|
        embed_dim = config.model.embed_dim      #  d 
        hidden_dim = config.model.hidden_dim    #  h

        # Store weight matrices in column vector notation, matching our mathematical formulation.
        # E_L : Left embedding associated with elements a         (d x |G|)
        # E_R : Right embedding associated with elements b        (d x |G|)
        # W_H : Hidden weight matrix                              (h x 2d)
        # W_U : Unembedding weight matrix                         (|G| x h)
        self.E_L = nn.Parameter(torch.randn(embed_dim, n)/np.sqrt(embed_dim))
        self.E_R = nn.Parameter(torch.randn(embed_dim, n)/np.sqrt(embed_dim))
        self.W_H = nn.Parameter(torch.randn(hidden_dim, 2*embed_dim)/np.sqrt(2*embed_dim))
        self.relu = nn.ReLU()
        self.W_U = nn.Parameter(torch.randn(n, hidden_dim)/np.sqrt(hidden_dim))


    def forward(self, input):
        n = self.E_L.shape[1]

        # One-hot encode inputs as column vectors
        e_a = F.one_hot(input[:, 0], num_classes=n).to(self.E_L.dtype).T
        e_b = F.one_hot(input[:, 1], num_classes=n).to(self.E_R.dtype).T

        # Embed the inputs
        a_embed = self.E_L @ e_a
        b_embed = self.E_R @ e_b

        # Concat to (2 * d) length column vector
        x = torch.cat((a_embed, b_embed), dim=0)
        
        # Pass to hidden layer
        x = self.W_H @ x
        
        # ReLU
        x = self.relu(x)

        # Unembed
        logits = self.W_U @ x

        # PyTorch's cross entropy expects shape (B, |G|) so we transpose
        return logits.T
