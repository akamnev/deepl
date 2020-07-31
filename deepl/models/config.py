import copy
import json
from enum import IntEnum, auto
import torch


class IntEnumBase(IntEnum):
    @classmethod
    def from_name(cls, name):
        try:
            return cls.__members__[name]
        except KeyError:
            cls._missing_(name)


class PSS(IntEnumBase):
    """Parameter Sharing Strategy"""
    NO_PARAMETERS_SHARING = auto()
    ALL_PARAMETERS_SHARING = auto()


class VPP(IntEnumBase):
    """Vector Place Position"""
    FIRST = auto()
    INSIDE = auto()
    LAST = auto()


class VAEType(IntEnumBase):
    """VAE encoder type"""
    NORMAL_TANH_ABS = auto()


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
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
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
        self.cross_layer_parameter_sharing = cross_layer_parameter_sharing
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        if not isinstance(self.cross_layer_parameter_sharing, PSS):
            self.cross_layer_parameter_sharing = PSS.from_name(
                self.cross_layer_parameter_sharing)

    def to_dict(self):
        output = super().to_dict()
        output['device'] = str(output['device'])
        output['cross_layer_parameter_sharing'] = self.cross_layer_parameter_sharing.name
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
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
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
                         cross_layer_parameter_sharing,
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
                 model_type=VPP.INSIDE,
                 device='cpu',
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 padding_idx=0,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
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
                         cross_layer_parameter_sharing,
                         use_cut_head,
                         ignore_index,
                         tie_embedding_vectors,
                         output_attentions,
                         output_hidden_states)
        self.model_type = model_type
        if not isinstance(self.model_type, VPP):
            self.model_type = VPP.from_name(self.model_type)


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
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
                 vae_type=VAEType.NORMAL_TANH_ABS,
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
                         cross_layer_parameter_sharing,
                         output_attentions,
                         output_hidden_states)
        self.vae_type = vae_type
        if not isinstance(self.vae_type, VAEType):
            self.vae_type = VAEType.from_name(VAEType)
