import torch.nn as nn
import torch

class GrokkingModel(nn.Module):
    def __init__(self, embed_dimension, hidden_dimension, n):
        super().__init__()

        self.left_embed = nn.Embedding(n, embed_dimension)
        self.right_embed = nn.Embedding(n, embed_dimension)
        self.hidden = nn.Linear(2*embed_dimension, hidden_dimension, bias=False)
        self.relu = nn.ReLU()
        self.unembedding = nn.Linear(hidden_dimension, n, bias=False)


    def forward(self, input):
        a_embed = self.left_embed(input[:, 0])
        b_embed = self.right_embed(input[:, 1])

        x = torch.cat((a_embed,b_embed), dim=1)
        x = self.hidden(x)
        x = self.relu(x)
        logits = self.unembedding(x)
        return logits