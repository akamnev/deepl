import numpy as np
import random
import torch


def random_token_sample(tokens, size=1, max_iter=None):
    if max_iter is None:
        max_iter = 10 * size
    ids, cnt = set(), 0
    while len(ids) < size and cnt < max_iter:
        ids.add(random.randrange(len(tokens)))
        cnt += 1
    return list(ids)


class InverseFreqSample:
    def __init__(self, token_count, alpha=0.5):
        self.alpha = alpha
        self.token_count = token_count

    def __call__(self, tokens, size=1, max_iter=None):
        if max_iter is None:
            max_iter = 10 * size
        proba = torch.tensor([self.token_count[w] for w in tokens],
                             dtype=torch.float32)
        proba = torch.pow(proba, -self.alpha)
        sampler = torch.distributions.categorical.Categorical(proba)
        ids, cnt = set(), 0
        while len(ids) < size and cnt < max_iter:
            for i in sampler.sample((size, )).tolist():
                ids.add(i)
                if len(ids) >= size:
                    break
            cnt += 1
        return list(ids)


def mask_token(tokens, id_mask, id_ignore,
               sampler=random_token_sample):
    sequence, labels = [], []
    for b in tokens:
        s = b[:]
        v = [id_ignore] * len(s)
        i = sampler(b, size=1)[0]
        v[i] = s[i]
        s[i] = id_mask
        sequence.append(s)
        labels.append(v)
    return sequence, labels


def mask_token_with_unity_mlm(tokens, proba_unity, proba_token,
                              id_mask, id_ignore, sampler=random_token_sample):
    sequence, labels, masked_pos = [], [], []
    for b in tokens:
        s = b[:]
        u = np.random.uniform()
        if u < proba_unity:
            v = s[:]
            i_to_mask = []
        else:
            v = [id_ignore] * len(s)
            num_token_to_mask = np.random.binomial(len(b), proba_token)
            num_token_to_mask = max(1, num_token_to_mask)
            num_token_to_mask = min(num_token_to_mask, len(s))
            i_to_mask = sampler(b, size=num_token_to_mask)
            for i in i_to_mask:
                v[i] = s[i]
                s[i] = id_mask
        sequence.append(s)
        labels.append(v)
        masked_pos.append(i_to_mask)
    return sequence, labels, masked_pos


def mask_token_with_unity_lm(tokens, proba_unity,
                             id_bos, id_eos, id_ignore,
                             sampler=random_token_sample):
    sequence, labels = [], []
    for b in tokens:
        u = np.random.uniform()
        if u < proba_unity:
            s = [id_bos] + b
            v = b + [id_eos]
        else:
            n = sampler(b, size=1)[0]
            s = [id_bos] + b[:n]
            v = [id_ignore] * len(s)
            v[n] = b[n]
        sequence.append(s)
        labels.append(v)
    return sequence, labels
