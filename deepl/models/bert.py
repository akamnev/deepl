"""Bidirectional Encoder Representations from Transformers"""
import torch
from .base import ModelBase
from ..layers.embeddings import (WordEmbeddings,
                                 VectorFirstEmbeddings,
                                 VectorLastEmbeddings,
                                 VectorInsideEmbeddings)
from ..layers.encoders import BertEncoder
from ..layers.headers import get_head_by_config
from ..layers.utils import get_attention_mask, get_vector_attention_mask
from .config import LanguageModelConfig, VPP


class LanguageModelBase(ModelBase):
    config_cls = LanguageModelConfig

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        self.encoder = BertEncoder(
            num_hidden_layers=config.encoder.num_hidden_layers,
            num_attention_heads=config.encoder.num_attention_heads,
            hidden_size=config.encoder.hidden_size,
            intermediate_size=config.encoder.intermediate_size,
            half_width_key=config.encoder.half_width_key,
            half_width_val=config.encoder.half_width_val,
            is_decoder=config.encoder.is_decoder,
            dropout_alpha=config.encoder.dropout_alpha,
            attention_head_size=config.encoder.attention_head_size,
            attention_type=config.encoder.attention_type,
            hidden_act=config.encoder.hidden_act,
            layer_norm_eps=config.encoder.layer_norm_eps,
            output_attentions=config.encoder.output_attentions,
            output_hidden_states=config.encoder.output_hidden_states)
        self.heads = torch.nn.ModuleDict(
            {name: get_head_by_config(cfg) for name, cfg in config.heads.items()})


class LanguageModel(LanguageModelBase):

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        self.embedding = WordEmbeddings(
            vocab_size=config.embeddings.vocab_size,
            hidden_size=config.embeddings.hidden_size,
            max_position=config.embeddings.max_position,
            device=config.embeddings.device,
            padding_idx=config.embeddings.padding_idx)

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs):
        hidden_states = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=hidden_states.device)
        outputs = self.encoder(hidden_states,
                               attention_mask,
                               encoder_hidden_states,
                               encoder_attention_mask)
        outputs = {
            'embeddings': outputs[0],
            'self_attentions': outputs[1],
            'hidden_states': outputs[2],
            'cross_attentions': outputs[3],
        }
        exclude_heads = kwargs.get('exclude_heads', set())
        for name, head in self.heads.items():
            if exclude_heads and name in exclude_heads:
                continue
            outputs[name] = head(
                embedding=outputs['embeddings'],
                attention_mask=attention_mask,
                **kwargs
            )
        return outputs


class VectorLanguageModel(LanguageModelBase):

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        if config.embeddings.model_type == VPP.FIRST:
            embeddings = VectorFirstEmbeddings
        elif config.embeddings.model_type == VPP.LAST:
            embeddings = VectorLastEmbeddings
        elif config.embeddings.model_type == VPP.INSIDE:
            embeddings = VectorInsideEmbeddings
        else:
            raise ValueError(f'{config.embeddings.model_type} not recognized. '
                             f'`model_type` should be set to either '
                             f'`FIRST`, `LAST`, or `INSIDE`')
        self.embedding = embeddings(
            vocab_size=config.embeddings.vocab_size,
            hidden_size=config.embeddings.hidden_size,
            max_position=config.embeddings.max_position,
            device=config.embeddings.device,
            padding_idx=config.embeddings.padding_idx)

    def forward(self,
                input_ids,
                vectors,
                attention_mask=None,
                input_pos=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs):

        hidden_states = self.embedding(
            input_ids=input_ids,
            vectors=vectors,
            input_pos=input_pos)

        if attention_mask is None:
            if self.config.embeddings.model_type == VPP.INSIDE:
                attention_mask = get_attention_mask(input_ids)
            else:
                attention_mask = get_vector_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)

        outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask)

        outputs = {
            'embeddings': outputs[0],
            'self_attentions': outputs[1],
            'hidden_states': outputs[2],
            'cross_attentions': outputs[3]
        }
        exclude_heads = kwargs.get('exclude_heads', set())
        for name, head in self.heads.items():
            if exclude_heads and name in exclude_heads:
                continue
            outputs[name] = head(
                embedding=outputs['embeddings'],
                attention_mask=attention_mask,
                **kwargs
            )
        return outputs
