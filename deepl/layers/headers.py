import torch
import torch.nn as nn
from torch.utils import checkpoint
from .activations import get_activation
from .dropout import VariationalNormalEpanechnikovDropout
from ..models.config import LanguageHeadConfig, VectorMeanHeadConfig, \
    VectorMaxHeadConfig, LinRegHeadConfig, LanguageHeadLNConfig, \
    VectorMeanLNHeadConfig, LanguageHeadSmallConfig

USE_CHECKPOINT = True


def get_head_by_config(config):
    if isinstance(config, LanguageHeadConfig):
        return LanguageModelHead(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            vocab_size=config.vocab_size,
            scale=config.scale
        )
    if isinstance(config, LanguageHeadSmallConfig):
        return LanguageModelSmallHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            scale=config.scale
        )
    elif isinstance(config, LanguageHeadLNConfig):
        return LanguageModelLNHead(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            vocab_size=config.vocab_size,
            layer_norm_eps=config.layer_norm_eps,
            scale=config.scale
        )
    elif isinstance(config, VectorMeanHeadConfig):
        return VectorMeanHead()
    elif isinstance(config, VectorMeanLNHeadConfig):
        return VectorMeanLNHead(
            hidden_size=config.hidden_size,
            layer_norm_eps=config.layer_norm_eps
        )
    elif isinstance(config, VectorMaxHeadConfig):
        return VectorMaxHead()
    elif isinstance(config, LinRegHeadConfig):
        return LinRegHead(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            output_size=config.output_size
        )
    else:
        raise ValueError(config)


class HeadBase(nn.Module):
    pass


class LanguageModelHead(HeadBase):

    def __init__(self, hidden_size, hidden_act, vocab_size, scale=0.1):
        super().__init__()
        self.scale = scale
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_dropout = VariationalNormalEpanechnikovDropout(hidden_size)
        self.act = get_activation(hidden_act)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        dense = torch.eye(self.hidden_size)
        dense += torch.rand((self.hidden_size, self.hidden_size)) / self.hidden_size
        self.dense.weight.data = dense

        d = self.decoder.weight.data
        d /= torch.std(d, dim=-1, keepdim=True)

    def forward(self, embedding, attention_mask, **kwargs):
        if 'labels_mask' in kwargs:
            labels_mask = kwargs['labels_mask']
            attention_mask = None
            embedding = embedding[labels_mask]
        hidden = self.dense(embedding)
        hidden = self.dense_dropout(hidden, attention_mask)
        if USE_CHECKPOINT:
            hidden = checkpoint.checkpoint(self.act, hidden)
        else:
            hidden = self.act(hidden)
        scores = self.scale * self.decoder(hidden)
        return scores


class LanguageModelSmallHead(HeadBase):
    def __init__(self, hidden_size, vocab_size, scale):
        super().__init__()
        self.scale = scale
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, embedding, **kwargs):
        if 'labels_mask' in kwargs:
            labels_mask = kwargs['labels_mask']
            embedding = embedding[labels_mask]
        scores = self.scale * self.decoder(embedding)
        return scores


class LanguageModelLNHead(LanguageModelHead):
    def __init__(self, hidden_size, hidden_act, vocab_size, scale, layer_norm_eps=1e-8):
        super().__init__(hidden_size, hidden_act, vocab_size, scale=scale)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps)

    def forward(self, embedding, attention_mask, **kwargs):
        embedding = self.layer_norm(embedding)
        return super().forward(embedding, attention_mask, **kwargs)


class VectorMeanHead(HeadBase):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self,
                embedding,
                attention_mask=None,
                **kwargs):
        norm = attention_mask.sum(dim=1, keepdim=True) + self.eps
        vectors = (embedding * attention_mask[..., None]).sum(dim=1) / norm
        return vectors


class VectorMeanLNHead(HeadBase):
    def __init__(self,
                 hidden_size,
                 layer_norm_eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.eps = 1e-8

    def forward(self,
                embedding,
                attention_mask=None,
                **kwargs):
        embedding = self.layer_norm(embedding)
        norm = attention_mask.sum(dim=1, keepdim=True) + self.eps
        vectors = (embedding * attention_mask[..., None]).sum(dim=1) / norm
        return vectors


class VectorMaxHead(HeadBase):
    def __init__(self):
        super().__init__()

    def forward(self,
                embedding,
                attention_mask=None,
                **kwargs):
        delta = embedding.detach().min() - embedding.detach().max() - 1.0
        vectors = embedding + (1.0 - attention_mask[..., None]) * delta
        vectors, _ = vectors.max(dim=1)
        return vectors


class LinRegHead(HeadBase):
    def __init__(self, hidden_size, hidden_act, output_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, output_size)
        self.dense_dropout = VariationalNormalEpanechnikovDropout(hidden_size)
        self.act = get_activation(hidden_act)

    def forward(self, embedding, attention_mask, **kwargs):
        hidden = self.dense_dropout(embedding, attention_mask)
        hidden = self.dense(hidden)
        scores = self.act(hidden)
        return scores
