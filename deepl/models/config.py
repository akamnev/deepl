import copy
import json
from enum import IntEnum, auto
import torch


class PSS(IntEnum):
    """Parameter Sharing Strategy"""
    NO_PARAMETERS_SHARING = auto()
    ALL_PARAMETERS_SHARING = auto()


class VPP(IntEnum):
    """Vector Place Position"""
    FIRST = auto()
    INSIDE = auto()
    LAST = auto()


class AttentionType(IntEnum):
    """Attention decoder type"""
    BIDIRECTIONAL = auto()
    AUTOREGRESSION = auto()


class GatingKind(IntEnum):
    NONE = auto()
    VectorGating = auto()
    ScalaGating = auto()


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
                 attention_half_width=None,
                 attention_type=AttentionType.BIDIRECTIONAL,
                 self_attention_bias=True,
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
        self.attention_half_width = attention_half_width
        self.attention_type = attention_type
        self.self_attention_bias = self_attention_bias
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if not isinstance(self.attention_type, AttentionType):
            self.attention_type = AttentionType[self.attention_type]

    def to_dict(self):
        output = super().to_dict()
        output['attention_type'] = str(self.attention_type.name)
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
                 max_position=0,
                 padding_idx=0,
                 device='cpu'):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position=max_position,
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
    def __init__(
            self,
            hidden_size,
            hidden_act,
            vocab_size
    ):
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


class VectorMeanLNHeadConfig(HeadConfigBase):
    """Голова для совместимости с предыдущей версией векторизатора"""
    def __init__(self,
                 hidden_size,
                 layer_norm_eps):
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps


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
        heads = heads if heads is not None else dict()
        for name, head in heads.items():
            if isinstance(head, dict):
                for cls in (LanguageHeadConfig,
                            LanguageHeadLNConfig,
                            LinRegHeadConfig,
                            VectorMeanHeadConfig,
                            VectorMeanLNHeadConfig,
                            VectorMaxHeadConfig):
                    if head['class_name'] == cls.__name__:
                        head = cls.from_dict(head)
                        break
            if not isinstance(head, (LanguageHeadConfig,
                                     LanguageHeadLNConfig,
                                     LinRegHeadConfig,
                                     VectorMeanHeadConfig,
                                     VectorMeanLNHeadConfig,
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


class SGWEmbeddingsConfig(ConfigBase):
    def __init__(
        self,
        workspace_size,
        vocab_size,
        workspace_hidden_size,
        token_hidden_size
    ):
        self.workspace_size = workspace_size
        self.vocab_size = vocab_size
        self.workspace_hidden_size = workspace_hidden_size
        self.token_hidden_size = token_hidden_size


class SGWDecoderEmbeddingsConfig(ConfigBase):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position = max_position


class SGWEncoderConfig(ConfigBase):
    def __init__(
        self,
        workspace_size,
        num_hidden_layers,
        workspace_hidden_size,
        token_hidden_size,
        num_workspace_attention_heads,
        num_token_attention_heads,
        intermediate_size,
        attention_half_width,
        hidden_act='ReLU',
        gating_h2m=GatingKind.NONE,
        gating_m2h=GatingKind.NONE,
        max_position=None,
        layer_norm_eps=1e-8,
        use_local_self_attention=True
    ):
        self.workspace_size = workspace_size
        self.num_hidden_layers = num_hidden_layers
        self.workspace_hidden_size = workspace_hidden_size
        self.token_hidden_size = token_hidden_size
        self.num_workspace_attention_heads = num_workspace_attention_heads
        self.num_token_attention_heads = num_token_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_half_width = attention_half_width
        self.hidden_act = hidden_act
        self.gating_h2m = gating_h2m
        self.gating_m2h = gating_m2h
        self.max_position = max_position
        self.layer_norm_eps = layer_norm_eps
        self.use_local_self_attention = use_local_self_attention

        if not isinstance(self.gating_h2m, GatingKind):
            self.gating_h2m = GatingKind[self.gating_h2m]
        if not isinstance(self.gating_m2h, GatingKind):
            self.gating_m2h = GatingKind[self.gating_m2h]

    def to_dict(self):
        output = super().to_dict()
        output['gating_h2m'] = str(self.gating_h2m.name)
        output['gating_m2h'] = str(self.gating_m2h.name)
        return output


class SGWDecoderConfig(ConfigBase):
    def __init__(
        self,
        num_hidden_layers,
        encoder_hidden_size,
        token_hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_act='ReLU',
        layer_norm_eps=1e-8
    ):
        self.num_hidden_layers = num_hidden_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.token_hidden_size = token_hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


class SGWLanguageModelConfig(ConfigBase):
    def __init__(
            self,
            embeddings,
            encoder,
            heads=None
    ):

        if isinstance(embeddings, dict):
            embeddings = SGWEmbeddingsConfig.from_dict(embeddings)
        self.embeddings = embeddings
        if not isinstance(self.embeddings,
                          (
                              SGWEmbeddingsConfig,
                              SGWDecoderEmbeddingsConfig
                          )):
            raise ValueError(self.embeddings)

        if isinstance(encoder, dict):
            encoder = SGWEncoderConfig.from_dict(encoder)
        if isinstance(encoder,
                      (
                          SGWEncoderConfig,
                          SGWDecoderConfig
                      )):
            self.encoder = encoder
        else:
            raise ValueError(encoder)

        self.heads = {}
        heads = heads if heads is not None else dict()
        for name, head in heads.items():
            if isinstance(head, dict):
                for cls in (LanguageHeadConfig,
                            LanguageHeadLNConfig,
                            LinRegHeadConfig,
                            VectorMeanHeadConfig,
                            VectorMeanLNHeadConfig,
                            VectorMaxHeadConfig):
                    if head['class_name'] == cls.__name__:
                        head = cls.from_dict(head)
                        break
            if not isinstance(head, (LanguageHeadConfig,
                                     LanguageHeadLNConfig,
                                     LinRegHeadConfig,
                                     VectorMeanHeadConfig,
                                     VectorMeanLNHeadConfig,
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
