import torch
from .base import ModelBase
from .config import SGWLanguageModelConfig
from ..layers.sgw import Embeddings, Encoder
from ..layers.headers import get_head_by_config


class SGWLanguageModel(ModelBase):
    config_cls = SGWLanguageModelConfig

    def __init__(self, config: SGWLanguageModelConfig):
        super().__init__(config)
        self.embedding = Embeddings(
            workspace_size=config.embeddings.workspace_size,
            vocab_size=config.embeddings.vocab_size,
            hidden_size=config.embeddings.hidden_size
        )
        self.encoder = Encoder(
            num_hidden_layers=config.encoder.num_hidden_layers,
            hidden_size=config.encoder.hidden_size,
            num_attention_heads=config.encoder.num_attention_heads,
            intermediate_size=config.encoder.intermediate_size,
            attention_half_width=config.encoder.attention_half_width,
            hidden_act=config.encoder.hidden_act,
            gating=config.encoder.gating,
            layer_norm_eps=config.encoder.layer_norm_eps
        )
        self.heads = torch.nn.ModuleDict(
            {name: get_head_by_config(cfg) for name, cfg in config.heads.items()}
        )

    def forward(
            self,
            input_ids,
            attention_mask,
            **kwargs
    ):
        workspace, embedding = self.embedding(input_ids, attention_mask)
        outputs = self.encoder(
            workspace_states=workspace,
            hidden_states=embedding,
            attention_mask=attention_mask,
            n_layer=kwargs.get('n_layer', None),
            output_hidden_states=kwargs.get('output_hidden_states', False)
        )
        outputs = {
            'workspace': outputs[0],
            'embedding': outputs[1],
            'all_workspace_states': outputs[2],
            'all_hidden_states': outputs[3],
        }
        exclude_heads = kwargs.get('exclude_heads', set())
        for name, head in self.heads.items():
            if exclude_heads and name in exclude_heads:
                continue
            outputs[name] = head(
                workspace=outputs['workspace'],
                embedding=outputs['embedding'],
                attention_mask=attention_mask,
                **kwargs
            )
        return outputs
