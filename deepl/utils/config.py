import copy
import json
import torch


class ConfigBase:
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.write(self.to_json())

    @classmethod
    def from_file(cls, file_path, **kwargs):
        with open(file_path, 'r') as fp:
            config = cls(**json.load(fp))

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def __repr__(self):
        return self.to_json()


class BERTConfig(ConfigBase):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 vocab_size,
                 max_position_embedding,
                 device='cpu',
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 padding_idx=0,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 output_attentions=False,
                 output_hidden_states=False):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embedding = max_position_embedding
        self.device = device
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.padding_idx = padding_idx
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def to_dict(self):
        output = super().to_dict()
        output['device'] = str(output['device'])
        return output


class BERTLanguageModelConfig(BERTConfig):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 vocab_size,
                 max_position_embedding,
                 device='cpu',
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 padding_idx=0,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 use_cut_head=True,
                 ignore_index=-100,
                 tie_embedding_vectors=True,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__(num_hidden_layers,
                         num_attention_heads,
                         hidden_size,
                         intermediate_size,
                         vocab_size,
                         max_position_embedding,
                         device,
                         dropout_prob,
                         layer_norm_eps,
                         is_decoder,
                         padding_idx,
                         hidden_act,
                         initializer_range,
                         output_attentions,
                         output_hidden_states)
        self.use_cut_head = use_cut_head
        self.ignore_index = ignore_index
        self.tie_embedding_vectors = tie_embedding_vectors


class VectorTextBERTConfig(BERTLanguageModelConfig):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 vocab_size,
                 max_position_embedding,
                 model_type='first',
                 device='cpu',
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 padding_idx=0,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 use_cut_head=True,
                 ignore_index=-100,
                 tie_embedding_vectors=True,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__(num_hidden_layers,
                         num_attention_heads,
                         hidden_size,
                         intermediate_size,
                         vocab_size,
                         max_position_embedding,
                         device,
                         dropout_prob,
                         layer_norm_eps,
                         is_decoder,
                         padding_idx,
                         hidden_act,
                         initializer_range,
                         use_cut_head,
                         ignore_index,
                         tie_embedding_vectors,
                         output_attentions,
                         output_hidden_states)
        self.model_type = model_type


class TextVectorVAEConfig(BERTConfig):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 vocab_size,
                 max_position_embedding,
                 device='cpu',
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 padding_idx=0,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 vae_type=None,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__(num_hidden_layers,
                         num_attention_heads,
                         hidden_size,
                         intermediate_size,
                         vocab_size,
                         max_position_embedding,
                         device,
                         dropout_prob,
                         layer_norm_eps,
                         is_decoder,
                         padding_idx,
                         hidden_act,
                         initializer_range,
                         output_attentions,
                         output_hidden_states)
        self.vae_type = vae_type