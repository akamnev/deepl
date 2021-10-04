import torch.nn as nn
from .activations import get_activation
from .dropout import VariationalNormalEpanechnikovDropout
from ..models.config import LanguageHeadConfig, VectorMeanHeadConfig, \
    VectorMaxHeadConfig, LinRegHeadConfig, LanguageHeadLNConfig


def get_head_by_config(config):
    if isinstance(config, LanguageHeadConfig):
        return LanguageModelHead(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            vocab_size=config.vocab_size
        )
    elif isinstance(config, LanguageHeadLNConfig):
        return LanguageModelLNHead(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            vocab_size=config.vocab_size,
            layer_norm_eps=config.layer_norm_eps
        )
    elif isinstance(config, VectorMeanHeadConfig):
        return VectorMeanHead()
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

    def __init__(self, hidden_size, hidden_act, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_dropout = VariationalNormalEpanechnikovDropout(hidden_size)
        self.act = get_activation(hidden_act)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.decoder_dropout = VariationalNormalEpanechnikovDropout(vocab_size)

    def forward(self, embedding, attention_mask, **kwargs):
        if 'labels_mask' in kwargs:
            labels_mask = kwargs['labels_mask']
            attention_mask = None
            embedding = embedding[labels_mask]
        hidden = self.dense(embedding)
        hidden = self.dense_dropout(hidden, attention_mask)
        hidden = self.act(hidden)
        scores = self.decoder(hidden)
        scores = self.decoder_dropout(scores, attention_mask)
        return scores


class LanguageModelLNHead(LanguageModelHead):
    def __init__(self, hidden_size, hidden_act, vocab_size, layer_norm_eps=1e-8):
        super().__init__(hidden_size, hidden_act, vocab_size)
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
