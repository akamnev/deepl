import torch
import torch.nn as nn


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def get_min_value(tensor):
    if tensor.dtype == torch.float16:
        min_value = -1e4
    elif tensor.dtype == torch.float32:
        min_value = -1e9
    else:
        raise ValueError("{} not recognized. `dtype` "
                         "should be set to either `torch.float32` "
                         "or `torch.float16`".format(tensor.dtype))
    return min_value


def get_attention_mask(input_ids):
    max_length = max([len(x) for x in input_ids])
    attention_mask = [[1.0] * len(x) + [0.0] * (max_length - len(x))
                      for x in input_ids]
    return attention_mask


def prune_input_sequence(input_ids, max_length):
    fval = []
    for ids in input_ids:
        if len(ids) > max_length:
            ids = ids[:max_length]
        fval.append(ids)
    return fval


def kl_div(mu, sigma):
    """
    KL-divergence between a diagonal multivariate normal,
    and a standard normal distribution (with zero mean and unit variance)
    """
    sigma_2 = sigma * sigma
    kld = 0.5 * torch.mean(mu * mu + sigma_2 - torch.log(sigma_2) - 1.0)
    return kld


def rand_epanechnikov_trig(shape, device, dtype):
    # https://stats.stackexchange.com/questions/6643/what-is-the-closed-form-solution-for-the-inverse-cdf-for-epanechnikov
    xi = torch.rand(shape,
                    dtype=dtype,
                    device=device)
    xi = 2 * torch.sin(torch.asin(2 * xi - 1) / 3)
    return xi
