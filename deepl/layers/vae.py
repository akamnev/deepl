import torch
import torch.nn as nn


class VAENormalTanhAbs(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense_mu = nn.Linear(hidden_size, hidden_size)
        self.dense_sigma = nn.Linear(hidden_size, hidden_size)
        self.activation_mu = nn.Tanh()
        self.activation_sigma = torch.abs

    def forward(self, vectors):
        mu = self.activation_mu(self.dense_mu(vectors))
        sigma = self.activation_sigma(self.dense_sigma(vectors))
        return mu, sigma
