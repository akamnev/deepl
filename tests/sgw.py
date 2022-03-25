import pytest
import random
import torch
from deepl.layers.sgw import LocalSelfAttention, SharedWorkSpace, \
    EncoderLayer, Encoder, Embeddings
from deepl.models.config import GatingKind, SGWEncoderConfig, \
    SGWEmbeddingsConfig, SGWLanguageModelConfig, LanguageHeadConfig
from deepl.models.sgw import SGWLanguageModel


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
    _, h, m, _, hidden_size, head_number, _, hw = input_tensors
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


def test_embedding(input_ids):
    ids, m, workspace_size, vocab_size, ws_hidden_size, hidden_size, _, _, _, _ = input_ids
    nm = torch.ones_like(m, dtype=torch.bool)
    nm[0, 0] = False
    obj = Embeddings(
        workspace_size=workspace_size,
        vocab_size=vocab_size,
        workspace_hidden_size=ws_hidden_size,
        token_hidden_size=hidden_size
    )

    output = obj(
        input_ids=ids,
        attention_mask=m,
        normalize_mask=nm
    )
    loss = obj.loss_norm_emb()
    print(output)
    print(loss)


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
