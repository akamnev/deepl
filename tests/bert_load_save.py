import torch
from deepl.models.bert import LanguageModel
from deepl.models.config import EncoderConfig, WordEmbeddingsConfig, \
    LanguageHeadConfig, VectorMeanHeadConfig, VectorMaxHeadConfig, \
    LanguageModelConfig


def create_model_config(is_decoder=False):
    cfg_encoder = EncoderConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=10,
        intermediate_size=100,
        is_decoder=is_decoder,
        output_attentions=True,
        output_hidden_states=True
    )

    cfg_embedding = WordEmbeddingsConfig(
        vocab_size=100,
        hidden_size=10,
        max_position=12
    )

    cfg_lm = LanguageHeadConfig(
        hidden_size=10,
        hidden_act='ReLU',
        vocab_size=100
    )
    cfg_mean = VectorMeanHeadConfig()
    cfg_max = VectorMaxHeadConfig()

    cfg_model = LanguageModelConfig(
        embeddings=cfg_embedding,
        encoder=cfg_encoder,
        heads={
            'tokens': cfg_lm,
            'mean_vector': cfg_mean,
            'max_vector': cfg_max
        }
    )
    return cfg_model


def test_serialize_deserialize():
    cfg = create_model_config(is_decoder=True)
    d = cfg.to_dict()
    LanguageModelConfig.from_dict(d)


def test_model_create():
    cfg = create_model_config(is_decoder=True)
    model = LanguageModel(cfg)
    for k in model.state_dict():
        print(k)


def test_model_run():
    cfg = create_model_config(is_decoder=True)
    model = LanguageModel(cfg)
    enc_v = torch.rand((2, 8, 10))
    enc_attn = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    input_ids = [list(range(5)), list(range(5))]
    outputs = model(
        input_ids=input_ids,
        encoder_hidden_states=enc_v,
        encoder_attention_mask=enc_attn)


def test_model_lm_run():
    cfg = create_model_config(is_decoder=False)
    model = LanguageModel(cfg)
    input_ids = [list(range(5)), list(range(5))]
    attention_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0, 1.0]])
    labels_mask = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0]])
    labels_mask = labels_mask > 0.0

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels_mask=labels_mask
    )


