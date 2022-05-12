import torch
from deepl.utils.init import gaussian_random_projection


def test_gaussian_random_projection():
    n, m = 8, 32
    rp = gaussian_random_projection(n, m)

    assert rp.shape == (n, m)

    norm = torch.norm(rp, dim=-1)
    assert torch.all(torch.isclose(norm, torch.tensor(1.0)))

    for i in range(n):
        for j in range(i + 1, n):
            s = rp[i] @ rp[j].T
            assert torch.isclose(s, torch.tensor(0.0), atol=1e-5)
