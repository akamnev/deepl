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
            workspace_hidden_size=config.embeddings.workspace_hidden_size,
            token_hidden_size=config.embeddings.token_hidden_size
        )
        self.encoder = Encoder(
            workspace_size=config.encoder.workspace_size,
            num_hidden_layers=config.encoder.num_hidden_layers,
            workspace_hidden_size=config.encoder.workspace_hidden_size,
            token_hidden_size=config.encoder.token_hidden_size,
            num_workspace_attention_heads=config.encoder.num_workspace_attention_heads,
            num_token_attention_heads=config.encoder.num_token_attention_heads,
            intermediate_size=config.encoder.intermediate_size,
            attention_half_width=config.encoder.attention_half_width,
            hidden_act=config.encoder.hidden_act,
            gating_h2m=config.encoder.gating_h2m,
            gating_m2h=config.encoder.gating_m2h,
            max_position=config.encoder.max_position,
            layer_norm_eps=config.encoder.layer_norm_eps,
            use_local_self_attention=config.encoder.use_local_self_attention
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
        workspace, embedding = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            avg_token_mix=kwargs.get('avg_token_mix', None)
        )
        outputs = self.encoder(
            workspace_states=workspace,
            hidden_states=embedding,
            attention_mask=attention_mask,
            n_layer=kwargs.get('n_layer', None),
            output_hidden_states=kwargs.get('output_hidden_states', False),
            output_proba=kwargs.get('output_proba', False),
            output_regularisation=kwargs.get('output_regularisation', False),
        )
        outputs = {
            'workspace': outputs[0],
            'embedding': outputs[1],
            'all_workspace_states': outputs[2],
            'all_hidden_states': outputs[3],
            'all_proba_lsa': outputs[4],
            'all_proba_ws_h2m': outputs[5],
            'all_proba_ws_m2h': outputs[6],
            'all_gating_h2m': outputs[7],
            'all_gating_m2h': outputs[8],
            'all_reg_sigma_arg': outputs[9],
            'all_reg_diff_norm': outputs[10],
            'all_sgw_value_unity': outputs[11]
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
