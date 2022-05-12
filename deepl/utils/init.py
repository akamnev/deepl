import torch


def gaussian_random_projection(n, m, device='cpu'):
    rp = []
    for i in range(n):
        r = torch.randn((1, m), device=device)
        r /= torch.norm(r, dim=-1)
        for j in range(i):
            r -= (r @ rp[j].T) * rp[j]
            r /= torch.norm(r, dim=-1)
        rp.append(r)
    rp = torch.cat(rp)
    return rp
