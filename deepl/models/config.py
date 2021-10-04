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


class AttentionType(IntEnumBase):
    """Attention decoder type"""
    BIDIRECTIONAL = auto()
    AUTOREGRESSION = auto()


def set_config_attrib(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)
    for child in dir(obj):
        child = getattr(obj, child)
        if isinstance(child, ConfigBase):
            set_config_attrib(child, name, value)
        elif isinstance(child, (list, tuple, set)):
            for ch in child:
                set_config_attrib(ch, name, value)
        elif isinstance(child, dict):
            for ch in child.values():
                set_config_attrib(ch, name, value)


class ConfigBase:
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if 'device' in output:
            output['device'] = str(output['device'])
        output['class_name'] = self.__class__.__name__
        return output

    def to_json(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.write(self.to_json())

    @classmethod
    def from_dict(cls, inputs):
        if 'class_name' in inputs:
            inputs.pop('class_name')
        return cls(**inputs)

    @classmethod
    def from_file(cls, file_path, **kwargs):
        with open(file_path, 'r') as fp:
            config = cls.from_dict(json.load(fp))

        for key, value in kwargs.items():
            set_config_attrib(config, key, value)
        return config

    def __repr__(self):
        return self.to_json()


class EncoderConfig(ConfigBase):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 half_width_key=0,
                 half_width_val=0,
                 is_decoder=False,
                 device='cpu',
                 dropout_alpha=0.0,
                 attention_head_size=None,
                 layer_norm_eps=1e-8,
                 padding_idx=0,
                 hidden_act='ReLU',
                 attention_type=AttentionType.BIDIRECTIONAL,
                 output_attentions=False,
                 output_hidden_states=False):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.half_width_key = half_width_key
        self.half_width_val = half_width_val
        self.is_decoder = is_decoder
        self.device = device
        self.dropout_alpha = dropout_alpha
        self.attention_head_size = attention_head_size
        self.layer_norm_eps = layer_norm_eps
        self.padding_idx = padding_idx
        self.hidden_act = hidden_act
        self.attention_type = attention_type
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if not isinstance(self.attention_type, AttentionType):
            self.attention_type = AttentionType.from_name(self.attention_type)

    def to_dict(self):
        output = super().to_dict()
        output['attention_type'] = self.attention_type.name
        return output


class EmbeddingsConfigBase(ConfigBase):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position=0,
                 padding_idx=0,
                 device='cpu'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position = max_position
        self.padding_idx = padding_idx
        self.device = device


class WordEmbeddingsConfig(EmbeddingsConfigBase):
    pass


class VectorEmbeddingsConfig(EmbeddingsConfigBase):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 model_type,
                 padding_idx=0,
                 device='cpu'):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            device=device)
        self.model_type = model_type

        if not isinstance(self.model_type, VPP):
            self.model_type = VPP.from_name(self.model_type)

    def to_dict(self):
        output = super().to_dict()
        output['model_type'] = self.model_type.name
        return output


class HeadConfigBase(ConfigBase):
    pass


class LanguageHeadConfig(HeadConfigBase):
    def __init__(self,
                 hidden_size,
                 hidden_act,
                 vocab_size):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size


class LanguageHeadLNConfig(HeadConfigBase):
    def __init__(self,
                 hidden_size,
                 hidden_act,
                 vocab_size,
                 layer_norm_eps):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps


class VectorMeanHeadConfig(HeadConfigBase):
    pass


class VectorMaxHeadConfig(HeadConfigBase):
    pass


class LinRegHeadConfig(HeadConfigBase):
    def __init__(self,
                 hidden_size,
                 hidden_act,
                 output_size):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.output_size = output_size


class LanguageModelConfig(ConfigBase):
    def __init__(self,
                 embeddings,
                 encoder,
                 heads=None):

        if isinstance(embeddings, dict):
            if embeddings['class_name'] == WordEmbeddingsConfig.__name__:
                embeddings = WordEmbeddingsConfig.from_dict(embeddings)
            elif embeddings['class_name'] == VectorEmbeddingsConfig.__name__:
                embeddings = VectorEmbeddingsConfig.from_dict(embeddings)
        self.embeddings = embeddings
        if not isinstance(self.embeddings, (WordEmbeddingsConfig,
                                            VectorEmbeddingsConfig)):
            raise ValueError(self.embeddings)

        if isinstance(encoder, dict):
            encoder = EncoderConfig.from_dict(encoder)
        if isinstance(encoder, EncoderConfig):
            self.encoder = encoder
        else:
            raise ValueError(encoder)

        self.heads = {}
        for name, head in heads.items():
            if isinstance(head, dict):
                for cls in (LanguageHeadConfig,
                            LanguageHeadLNConfig,
                            LinRegHeadConfig,
                            VectorMeanHeadConfig,
                            VectorMaxHeadConfig):
                    if head['class_name'] == cls.__name__:
                        head = cls.from_dict(head)
                        break
            if not isinstance(head, (LanguageHeadConfig,
                                     LanguageHeadLNConfig,
                                     LinRegHeadConfig,
                                     VectorMeanHeadConfig,
                                     VectorMaxHeadConfig)):
                raise ValueError(head)
            self.heads[name] = head

    def to_dict(self):
        outputs = {
            'embeddings': self.embeddings.to_dict(),
            'encoder': self.encoder.to_dict(),
            'heads': {k: h.to_dict() for k, h in self.heads.items()}
        }
        return outputs
