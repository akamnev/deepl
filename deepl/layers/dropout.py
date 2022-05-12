import torch
import torch.nn as nn
from typing import Optional
from scipy import special
import math
from .utils import kld_gaussian, rand_epanechnikov_trig


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


class VariationalBase(nn.Module):
    def __init__(self, input_size, momentum=0.99, eps=1e-8):
        super().__init__()
        self.input_size = input_size
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.zeros(input_size))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.num_batches_tracked.zero_()

    @torch.jit.unused
    def update(
        self,
        input_vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        input_vector = input_vector.view(-1, self.input_size)
        if mask is not None:
            mask = mask.view(-1)
            input_vector = input_vector[mask > 0]
        mean = input_vector.data.mean(dim=0)
        self.running_mean.data.mul_(self.momentum).add_(mean, alpha=1-self.momentum)
        std = input_vector.data.std(dim=0)
        self.running_var.data.mul_(self.momentum).addcmul_(std, std, value=1-self.momentum)
        self.num_batches_tracked.data.add_(1)

    @property
    def _correction(self):
        return 1.0 - self.momentum ** self.num_batches_tracked

    @property
    def mean(self):
        return self.running_mean / self._correction

    @property
    def var(self):
        return self.running_var / self._correction

    @property
    def snr(self):
        return self.mean / (torch.sqrt(self.var) + self.eps)


class VariationalGaussianDropout(VariationalBase):
    """Вариационный слой регуляризации с априорным и апостериорным
    нормальными распределениями
    """
    def __init__(self, input_size, truncate=None, momentum=0.99, eps=1e-8):
        super().__init__(input_size, momentum=momentum, eps=eps)
        self.truncate = truncate
        self.log_sigma = nn.Parameter(torch.Tensor(input_size))
        self.log_sigma.data.fill_(-1.0)
        self._mean = None

    def forward(
        self,
        vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        epsilon = torch.randn(vector.size(), device=vector.device)
        if self.truncate is not None:
            epsilon = torch.fmod(epsilon, self.truncate)
        if mask is not None:
            epsilon = epsilon * mask[..., None]
        variance = torch.exp(self.log_sigma)
        if self.training:
            self._save_stat(vector)

        vector = vector + variance * epsilon
        if self.training:
            self.update(vector, mask)
        return vector

    @torch.jit.unused
    def _save_stat(self, vector):
        self._mean = vector

    def kld(self, nu=0.0, rho=1.0):
        return kld_gaussian(self._mean, self.log_sigma, nu=nu, rho=rho)


class VariationalNormalEpanechnikovDropout(VariationalBase):
    def __init__(self, input_size, momentum=0.99, eps=1e-8):
        super().__init__(input_size, momentum=momentum, eps=eps)
        self.log_sigma = nn.Parameter(torch.Tensor(input_size))
        self.log_sigma.data.fill_(-10.0)
        self._mean = None
        self._const = 0.5*math.log(90.0*math.pi) - 7./6.
        self._shift = 0.5*math.log(5.0)

    def forward(
        self,
        vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):

        epsilon = rand_epanechnikov_trig(vector.size(), device=vector.device)
        if mask is not None:
            epsilon = epsilon * mask[..., None]
        variance = torch.exp(self.log_sigma)
        if self.training:
            self._save_stat(vector)

        vector = vector + variance * epsilon
        if self.training:
            self.update(vector, mask)
        return vector

    @torch.jit.unused
    def _save_stat(self, vector):
        self._mean = vector

    def kld(self, nu=0.0, rho=1.0):
        log_sigma = self.log_sigma - self._shift
        normal_kld = kld_gaussian(self._mean, log_sigma, nu=nu, rho=rho)
        return self._const + normal_kld


class VariationalLogNormalGammaDropout(VariationalBase):
    """Вариационный слой регуляризации с априорным гамма и апостериорным
    логнормальным распределениями
    """

    def __init__(self, input_size, truncate=None, momentum=0.99, eps=1e-8):
        super().__init__(input_size, momentum=momentum, eps=eps)
        self.truncate = truncate
        self.sigma = nn.Parameter(torch.Tensor(input_size))
        self.sigma.data.fill_(0.01)
        self.eps = eps
        self._mean = None
        self._coeff = None

    def forward(self, vector, mask=None):

        epsilon = torch.randn(vector.size(), device=vector.device)
        if self.truncate is not None:
            epsilon = torch.fmod(epsilon, self.truncate)
        xi = -0.5 * self.sigma * self.sigma + torch.abs(self.sigma) * epsilon

        if self.training:
            self._save_stat(vector)

        if mask is not None:
            xi = xi * mask[..., None]

        vector = vector * torch.exp(xi)
        if self.training:
            self.update(vector, mask)
        return vector

    def kld(self, alpha=0.01, beta=0.1):
        alpha, beta = float(alpha), float(beta)
        if self._coeff is None:
            self._coeff = {(alpha, beta): self._kld_coeff(alpha, beta)}
        elif (alpha, beta) not in self._coeff:
            self._coeff[(alpha, beta)] = self._kld_coeff(alpha, beta)
        const_coeff = self._coeff[(alpha, beta)]
        var_part = - torch.log(torch.abs(self.sigma) + self.eps) + 0.5 * alpha * self.sigma**2
        mean_part = beta * torch.abs(self._mean) - alpha * torch.log(torch.abs(self._mean) + self.eps)
        fval = const_coeff * torch.numel(var_part) + torch.sum(var_part) + torch.sum(mean_part)
        return fval

    @staticmethod
    def _kld_coeff(alpha, beta):
        return special.loggamma(alpha) - alpha * math.log(beta) \
               - 0.5 * math.log(2 * math.pi) - 0.5

    @torch.jit.unused
    def _save_stat(self, vector):
        self._mean = vector

