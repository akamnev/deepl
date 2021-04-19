import torch
import torch.nn as nn
from .utils import kld_gaussian


class GaussianDropout(nn.Module):
    def __init__(self, p=0.0, truncate=None):
        super().__init__()
        self.alpha = p / (1.0 - p)
        self.truncate = truncate

    def forward(self, vector):
        if self.training and self.alpha > 0.0:
            epsilon = torch.randn(vector.size(), device=vector.device)
            if self.truncate is not None:
                epsilon = torch.fmod(epsilon, self.truncate)
            epsilon = self.alpha * epsilon + 1.0
            vector = vector * epsilon
        return vector


class VariationalGaussianDropout(nn.Module):
    def __init__(self, input_size, truncate=None):
        super().__init__()
        self.truncate = truncate
        self.log_sigma = nn.Parameter(torch.Tensor(input_size))
        self.log_sigma.data.fill_(-1.0)
        self._mean = None

    def forward(self, vector):
        if self.training:
            epsilon = torch.randn(vector.size(), device=vector.device)
            if self.truncate is not None:
                epsilon = torch.fmod(epsilon, self.truncate)
            variance = torch.exp(self.log_sigma)

            self._mean = vector

            vector = vector + variance * epsilon
        return vector

    def kld(self, nu=0.0, rho=1.0):
        return kld_gaussian(self._mean, self.log_sigma, nu=nu, rho=rho)
