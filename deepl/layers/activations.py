import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class LeakySwish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakySwish}(x) = x(\frac{1-\text{negative_slope}}{1+e^{-\text{sharpness} x}} + \text{negative_slope})

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        sharpness: Controls the sharp of switch from positive axis to negative one. Default: 10.0

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, negative_slope=1e-2, sharpness=10.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.sharpness = sharpness

    def forward(self, inputs):
        outputs = torch.sigmoid(self.sharpness * inputs)
        outputs = (1.0 - self.negative_slope) * outputs + self.negative_slope
        outputs = inputs * outputs
        return outputs


class LeakySoftPlus(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakySoftPlus}(x) = \frac{1}{\text{sharpness}}f(\text{sharpness} x)
        f(x) = \log(1+e^x) - \log(1+e^{-\text{negative_slope}x})

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        sharpness: Controls the sharp of switch from positive axis to negative one. Default: 10.0

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, negative_slope=1e-2, sharpness=10.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.sharpness = sharpness

    def forward(self, inputs):
        outputs = self.sharpness * inputs
        outputs = -F.logsigmoid(-outputs) + F.logsigmoid(self.negative_slope * outputs)
        outputs = (1/self.sharpness) * outputs
        return outputs


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "leakyReLU": torch.nn.LeakyReLU(),
    "ReLU": F.relu
}


def get_activation(activation):
    if isinstance(activation, str):
        return ACT2FN[activation]
    elif isinstance(activation, (list, tuple)):
        activation_name = activation[0]
        activation_params = activation[1]
        if activation_name == 'LeakyReLU':
            return nn.LeakyReLU(**activation_params)
        elif activation_name == 'LeakySwish':
            return LeakySwish(**activation_params)
        elif activation_name == 'LeakySoftPlus':
            return LeakySoftPlus(**activation_params)
        else:
            raise ValueError(activation_name)
    elif isinstance(activation, dict):
        if activation['name'] == 'LeakyReLU':
            return nn.LeakyReLU(**activation['params'])
        elif activation['name'] == 'LeakySwish':
            return LeakySwish(**activation['params'])
        elif activation['name'] == 'LeakySoftPlus':
            return LeakySoftPlus(**activation['params'])
        else:
            raise ValueError(activation)
    elif callable(activation):
        return activation
    else:
        raise ValueError(activation)
