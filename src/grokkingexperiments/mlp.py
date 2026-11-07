import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        n = config.task.n
        embed_dim = config.model.embed_dim
        hidden_dim = config.model.hidden_dim

        self.left_embed = nn.Embedding(n, embed_dim)
        self.right_embed = nn.Embedding(n, embed_dim)
        self.hidden = nn.Linear(2 * embed_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.unembedding = nn.Linear(hidden_dim, n, bias=False)


    def forward(self, input):
        a_embed = self.left_embed(input[:, 0])
        b_embed = self.right_embed(input[:, 1])

        x = torch.cat((a_embed,b_embed), dim=1)
        x = self.hidden(x)
        x = self.relu(x)
        logits = self.unembedding(x)
        return logits