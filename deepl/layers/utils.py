import torch


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


def kld_gaussian(mu, log_sigma, nu=0.0, rho=1.0):
    """
    KL-divergence between a diagonal multivariate normal,
    and a standard normal distribution
    """
    device = mu.device
    nu = torch.as_tensor(nu, device=device)
    rho = torch.as_tensor(rho, device=device)
    delta_variance = 2.0 * (log_sigma - torch.log(rho))
    variance_term = torch.sum(torch.exp(delta_variance) - delta_variance)
    mean_term = torch.sum((mu - nu) ** 2 / rho)
    return 0.5 * (mean_term + variance_term - 1.0)


def rand_epanechnikov_trig(shape, device, dtype=torch.float32):
    # https://stats.stackexchange.com/questions/6643/what-is-the-closed-form-solution-for-the-inverse-cdf-for-epanechnikov
    xi = torch.rand(shape,
                    dtype=dtype,
                    device=device)
    xi = 2 * torch.sin(torch.asin(2 * xi - 1) / 3)
    return xi
