import numpy as np
import random


def random_token_sample(tokens, *args):
    return random.randrange(len(tokens))


class InverseFreqSample:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, tokens, tokens_count, *args):
        tokens_count = np.asarray(tokens_count)
        tokens_count = 1.0 / tokens_count
        tokens_count = np.power(tokens_count, self.alpha)
        tokens_count /= np.sum(tokens_count)
        tokens_count = np.cumsum(tokens_count)
        tokens_count = np.insert(tokens_count, 0, 0.0)
        u = np.random.uniform()
        for i in range(len(tokens_count)-1):
            v1 = tokens_count[i]
            v2 = tokens_count[i+1]
            if v1 <= u < v2:
                return i
        return len(tokens_count) - 1


def mask_token(tokens, token_count, id_mask, id_ignore,
               sampler=random_token_sample):
    sequence, labels = [], []
    for b, c in zip(tokens, token_count):
        s = b[:]
        v = [id_ignore] * len(s)
        i = sampler(b, c)
        v[i] = s[i]
        s[i] = id_mask
        sequence.append(s)
        labels.append(v)
    return sequence, labels


def mask_token_with_unity_mlm(tokens, token_count, proba_unity, proba_token,
                              id_mask, id_ignore, sampler=random_token_sample):
    sequence, labels, masked_pos = [], [], []
    for b, c in zip(tokens, token_count):
        s = b[:]
        u = np.random.uniform()
        if u < proba_unity:
            v = s[:]
            i_to_mask = set()
        else:
            v = [id_ignore] * len(s)
            num_token_to_mask = np.random.binomial(len(b), proba_token)
            num_token_to_mask = max(1, num_token_to_mask)
            num_token_to_mask = min(num_token_to_mask, len(s))
            i_to_mask = set()
            while len(i_to_mask) < num_token_to_mask:
                i_to_mask.add(sampler(b, c))
            for i in i_to_mask:
                v[i] = s[i]
                s[i] = id_mask
        sequence.append(s)
        labels.append(v)
        masked_pos.append(list(i_to_mask))
    return sequence, labels, masked_pos


def mask_token_with_unity_lm(tokens, token_count, proba_unity,
                             id_sos, id_eos, id_ignore,
                             sampler=random_token_sample):
    sequence, labels = [], []
    for b, c in zip(tokens, token_count):
        u = np.random.uniform()
        if u < proba_unity:
            s = [id_sos] + b
            v = b + [id_eos]
        else:
            n = sampler(b, c)
            s = [id_sos] + b[:n]
            v = [id_ignore] * len(s)
            v[n] = b[n]
        sequence.append(s)
        labels.append(v)
    return sequence, labels
