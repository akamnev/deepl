import pytest
import random
import torch
from deepl.layers.sgw import LocalSelfAttention, SharedWorkSpace, \
    EncoderLayer, Encoder, Embeddings, AutoRegressiveGlobalSelfAttention, \
    DecoderLayer, Decoder, DecoderEmbeddings, GlobalCrossAttention
from deepl.models.config import GatingKind, SGWEncoderConfig, \
    SGWEmbeddingsConfig, SGWLanguageModelConfig, LanguageHeadConfig, \
    SGWDecoderEmbeddingsConfig, SGWDecoderConfig, SGWDecoderModelConfig, \
    LanguageHeadSmallConfig
from deepl.models.sgw import SGWLanguageModel, DecoderModel


def create_example(
        batch_size_range=(3, 3),
        ws_number_range=(1, 1),
        ws_hidden_size_range=(32, 32),
        token_number_range=(10, 10),
        head_number_range=(2, 2),
        layer_number_range=(3, 3),
        head_size_range=(8, 8),
        hw_range=(3, 3),
        device='cpu'
):
    batch_size = random.randint(*batch_size_range)
    workspace_number = random.randint(*ws_number_range)
    ws_hidden_size = random.randint(*ws_hidden_size_range)
    token_number = random.randint(*token_number_range)
    head_number = random.randint(*head_number_range)
    layer_number = random.randint(*layer_number_range)
    head_size = random.randint(*head_size_range)
    hw = random.randint(*hw_range)

    hidden_size = head_size * head_number

    ws = torch.rand(
        (batch_size, workspace_number, ws_hidden_size),
        requires_grad=True,
        device=device
    )
    h = torch.rand(
        (batch_size, token_number, hidden_size),
        requires_grad=True,
        device=device
    )
    m = torch.ones(
        (batch_size, token_number),
        requires_grad=False,
        device=device,
        dtype=torch.float32
    )
    if token_number > 2:
        for i in range(1, batch_size):
            j = random.randint(1, token_number-1)
            m[i, j:] = 0.0

    return ws, h, m, workspace_number, ws_hidden_size, hidden_size, head_number, layer_number, hw


def create_ids_example(
        batch_size_range=(3, 3),
        token_number_range=(10, 10),
        ws_size_range=(1, 1),
        ws_hidden_size_range=(32, 32),
        vocab_size_range=(512, 512),
        layer_number_range=(3, 3),
        head_number_range=(2, 2),
        head_size_range=(8, 8),
        hw_range=(3, 3),
        gating=None,
        device='cpu'
):
    batch_size = random.randint(*batch_size_range)
    token_number = random.randint(*token_number_range)
    workspace_size = random.randint(*ws_size_range)
    vocab_size = random.randint(*vocab_size_range)
    ws_hidden_size = random.randint(*ws_hidden_size_range)
    layer_number = random.randint(*layer_number_range)
    head_number = random.randint(*head_number_range)
    head_size = random.randint(*head_size_range)
    hw = random.randint(*hw_range)

    hidden_size = head_size * head_number

    if gating is None:
        gating = random.choice([v for v in GatingKind])

    ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, token_number),
        device=device,
    )

    m = torch.ones(
        (batch_size, token_number),
        requires_grad=False,
        device=device,
        dtype=torch.float32
    )
    if token_number > 2:
        for i in range(1, batch_size):
            j = random.randint(1, token_number-1)
            m[i, j:] = 0.0

    return ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, \
           layer_number, head_number, hw, gating


@pytest.fixture
def input_tensors():
    return create_example()


@pytest.fixture
def input_ids():
    return create_ids_example()


def test_local_self_attention(input_tensors):
    _, h, m, _, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    obj = LocalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=head_number,
        attention_half_width=hw,
    )
    output = obj(
        hidden_states=h,
        attention_mask=m
    )
    reg_1 = obj.loss_value_unity()
    reg_2 = obj.loss_attention_entropy()
    print(output)
    print(reg_1)
    print(reg_2)


def test_init_orth_local_self_attention(input_tensors):
    _, h, m, _, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    obj = LocalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=head_number,
        attention_half_width=hw,
    )

    vw = obj.output_layer.weight @ obj.value_projection.weight
    dvw = torch.diag(vw)
    m = torch.eye(vw.shape[0], dtype=torch.bool)
    ovw = vw[~m]

    assert torch.all(torch.isclose(dvw, torch.tensor(1.0)))
    assert torch.all(torch.isclose(ovw, torch.tensor(0.0), atol=1e-5))

    obj.output_layer.weight.data += 2.0

    assert not torch.any(torch.isclose(obj.value_projection.weight, obj.output_layer.weight.T))


def test_init_rand_proj_local_self_attention(input_tensors):
    _, h, m, _, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    scale = 0.01
    obj = LocalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=head_number,
        attention_half_width=hw,
        scale=scale
    )
    rp = obj.query_projection.weight.data
    norm = torch.norm(rp, dim=-1)
    s = rp @ rp.T
    assert torch.all(torch.isclose(norm, torch.tensor(scale)))
    assert torch.all(torch.isclose(torch.diag(s), torch.tensor(scale ** 2)))

    rp = obj.key_projection.weight.data
    norm = torch.norm(rp, dim=-1)
    s = rp @ rp.T
    assert torch.all(torch.isclose(norm, torch.tensor(scale)))
    assert torch.all(torch.isclose(torch.diag(s), torch.tensor(scale ** 2)))


def test_autoregressive_global_self_attention(input_tensors):
    _, h, m, _, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    obj = AutoRegressiveGlobalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=head_number
    )
    output = obj(
        hidden_states=h,
        attention_mask=m
    )
    reg_1 = obj.loss_value_unity()
    reg_2 = obj.loss_attention_entropy()
    print(output)
    print(reg_1)
    print(reg_2)


def test_workspace(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, _, hw = input_tensors
    obj = SharedWorkSpace(
        workspace_size=workspace_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
        num_workspace_attention_heads=head_number,
        num_token_attention_heads=head_number,
        gating_h2m=GatingKind.ScalaGating,
        gating_m2h=GatingKind.ScalaGating,
        max_position=512
    )
    m = torch.as_tensor(m, dtype=torch.bool)
    output = obj(
        workspace_states=ws,
        hidden_states=h,
        attention_mask=m
    )
    print(output)


def test_init_global_attention(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, _, hw = input_tensors
    obj = GlobalCrossAttention(
        hidden_size_out=ws_hidden_size,
        hidden_size_in=hidden_size,
        num_attention_heads=head_number,
        max_position=512,
        scale=0.01
    )

    vw = obj.value.weight.T @ obj.value.weight
    dvw = torch.diag(vw)
    m = torch.eye(vw.shape[0], dtype=torch.bool)
    ovw = vw[~m]

    assert torch.all(torch.isclose(dvw, torch.tensor(1.0)))
    assert torch.all(torch.isclose(ovw, torch.tensor(0.0), atol=1e-5))

    vw = obj.output_layer.weight.T @ obj.output_layer.weight
    dvw = torch.diag(vw)
    m = torch.eye(vw.shape[0], dtype=torch.bool)
    ovw = vw[~m]

    assert torch.all(torch.isclose(dvw, torch.tensor(1.0)))
    assert torch.all(torch.isclose(ovw, torch.tensor(0.0), atol=1e-5))



def test_encoder_layer(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, _, hw = input_tensors
    m = torch.as_tensor(m, dtype=torch.bool)
    swsu = SharedWorkSpace(
        workspace_size=workspace_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
        num_workspace_attention_heads=head_number,
        num_token_attention_heads=head_number,
        gating_h2m=GatingKind.ScalaGating,
        gating_m2h=GatingKind.ScalaGating,
        max_position=512
    )

    obj = EncoderLayer(
        hidden_size=hidden_size,
        num_attention_heads=head_number,
        intermediate_size=4*hidden_size,
        attention_half_width=hw,
        hidden_act='ReLU',
        shared_work_space_unit=swsu,
    )

    output = obj(
        workspace_states=ws,
        hidden_states=h,
        attention_mask=m
    )
    print(output)


def test_decoder_layer(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, _, hw = input_tensors
    m = torch.as_tensor(m, dtype=torch.bool)
    enc_h = torch.randn((h.shape[0], 2 * h.shape[1], 2 * hidden_size))
    enc_m = torch.ones((h.shape[0], 2 * h.shape[1]), dtype=torch.bool)
    obj = DecoderLayer(
        hidden_size=hidden_size,
        encoder_hidden_size=2*hidden_size,
        num_attention_heads=head_number,
        intermediate_size=4*hidden_size,
        hidden_act='ReLU',
    )

    output = obj(
        hidden_states=h,
        attention_mask=m,
        encoder_hidden_states=enc_h,
        encoder_attention_mask=enc_m
    )
    print(output)


def test_encoder(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    m = torch.as_tensor(m, dtype=torch.bool)
    obj = Encoder(
        workspace_size=workspace_size,
        num_hidden_layers=layer_number,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
        num_workspace_attention_heads=head_number,
        num_token_attention_heads=head_number,
        intermediate_size=4*hidden_size,
        attention_half_width=hw,
        hidden_act='ReLU',
        gating_h2m=GatingKind.ScalaGating,
        gating_m2h=GatingKind.ScalaGating,
        max_position=512
    )

    output = obj(
        workspace_states=ws,
        hidden_states=h,
        attention_mask=m,
        output_hidden_states=True,
        output_proba=True,
        output_regularisation=True
    )
    print(output)


def test_decoder(input_tensors):
    ws, h, m, workspace_size, ws_hidden_size, hidden_size, head_number, layer_number, hw = input_tensors
    m = torch.as_tensor(m, dtype=torch.bool)
    encoder_hidden_size = 2 * hidden_size
    enc_h = torch.randn((h.shape[0], 2 * h.shape[1], 2 * hidden_size))
    enc_m = torch.ones((h.shape[0], 2 * h.shape[1]), dtype=torch.bool)
    obj = Decoder(
        num_hidden_layers=layer_number,
        encoder_hidden_size=encoder_hidden_size,
        token_hidden_size=hidden_size,
        num_attention_heads=head_number,
        intermediate_size=4*hidden_size
    )

    output = obj(
        hidden_states=h,
        attention_mask=m,
        encoder_hidden_states=enc_h,
        encoder_attention_mask=enc_m,
        n_layer=None,
        output_hidden_states=True,
        output_proba=True,
        output_regularisation=True
    )
    print(output)


def test_encoder_embedding(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, _, _, _, _ = input_ids
    obj = Embeddings(
        workspace_size=workspace_size,
        vocab_size=vocab_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
    )

    output = obj(
        input_ids=ids,
        attention_mask=m
    )
    print(output)


def test_init_encoder_embedding(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, _, _, _, _ = input_ids
    obj = Embeddings(
        workspace_size=workspace_size,
        vocab_size=vocab_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
    )
    std_we = torch.std(obj.word_embeddings.weight, dim=-1, keepdim=True)
    std_ws = torch.std(obj.init_workspace, dim=-1, keepdim=True)
    assert torch.all(torch.isclose(std_ws, torch.tensor(1.0)))
    assert torch.all(torch.isclose(std_we, torch.tensor(1.0)))


def test_decoder_embedding(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, _, _, _, _ = input_ids
    obj = DecoderEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position=512
    )
    pos_ids = torch.randint(0, 512, ids.shape)
    output = obj(
        input_ids=ids,
        position_ids=pos_ids,
        attention_mask=m
    )
    print(output)


def test_init_decoder_embedding(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, _, _, _, _ = input_ids
    obj = DecoderEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position=512
    )
    std_we = torch.std(obj.word_embeddings.weight, dim=-1, keepdim=True)
    std_ps = torch.std(obj.position_embeddings.weight, dim=-1, keepdim=True)
    assert torch.all(torch.isclose(std_we, torch.tensor(1.0)))
    assert torch.all(torch.isclose(std_ps, torch.tensor(1.0)))


def test_language_model(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, \
        layer_number, head_number, hw, gating = input_ids
    embeddings = SGWEmbeddingsConfig(
        workspace_size=workspace_size,
        vocab_size=vocab_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size
    )
    encoder = SGWEncoderConfig(
        workspace_size=workspace_size,
        num_hidden_layers=layer_number,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size,
        num_workspace_attention_heads=head_number,
        num_token_attention_heads=head_number,
        intermediate_size=4 * hidden_size,
        attention_half_width=hw,
        hidden_act='leakyReLU',
        gating_h2m=gating,
        gating_m2h=gating,
        layer_norm_eps=1e-8
    )
    lm_head = LanguageHeadConfig(
        hidden_size=hidden_size,
        hidden_act='ReLU',
        vocab_size=vocab_size

    )
    config = SGWLanguageModelConfig(
        embeddings=embeddings,
        encoder=encoder,
        heads={
            'tokens': lm_head
        }
    )
    model = SGWLanguageModel(config=config)
    output = model(
        input_ids=ids,
        attention_mask=m,
        n_layer=None,
        output_hidden_states=False,
        output_proba=True,
        output_regularisation=True
    )
    print(output)


def test_decoder_model(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, \
        layer_number, head_number, hw, gating = input_ids
    embeddings = SGWDecoderEmbeddingsConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position=512
    )
    encoder_hidden_size = 2 * hidden_size
    encoder = SGWDecoderConfig(
        num_hidden_layers=layer_number,
        encoder_hidden_size=encoder_hidden_size,
        token_hidden_size=hidden_size,
        num_attention_heads=head_number,
        intermediate_size=4 * hidden_size,
        hidden_act='leakyReLU',
        layer_norm_eps=1e-8
    )
    lm_head = LanguageHeadSmallConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    config = SGWDecoderModelConfig(
        embeddings=embeddings,
        encoder=encoder,
        heads={
            'tokens': lm_head
        }
    )
    model = DecoderModel(config=config)

    enc_h = torch.randn((ids.shape[0], 2 * ids.shape[1], encoder_hidden_size))
    enc_m = torch.ones((ids.shape[0], 2 * ids.shape[1]), dtype=torch.bool)
    position_ids = torch.randint(0, 512, ids.shape)
    output = model(
        input_ids=ids,
        position_ids=position_ids,
        attention_mask=m,
        encoder_hidden_states=enc_h,
        encoder_attention_mask=enc_m,
        n_layer=None,
        output_hidden_states=False,
        output_proba=True,
        output_regularisation=True
    )
    print(output)
