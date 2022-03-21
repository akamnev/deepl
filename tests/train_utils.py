import pytest
import random
import torch
import torch.nn as nn
from deepl.train.utils import collect_layer_norm_weights


class TestModel(nn.Module):
    def __init__(
            self,
            hidden_size,
            vocab_size
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.lin_1 = nn.Linear(hidden_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)
        self.lin_reg = nn.Linear(hidden_size, 1)
        self.act_1 = nn.LeakyReLU()
        self.act_2 = nn.LeakyReLU()

    def forward(self, ids):
        h = self.word_embeddings(ids)
        h = self.layer_norm_1(h)
        h = self.lin_1(h)
        h = self.act_1(h)
        h = self.layer_norm_2(h)
        h = self.lin_2(h)
        h = self.act_2(h)
        v = torch.mean(h, dim=1)
        x = self.lin_reg(v)
        return x


def create_ids_example(
        batch_size_range=(3, 3),
        token_number_range=(10, 10),
        vocab_size_range=(512, 512),
        hidden_size_range=(16, 16),
):
    batch_size = random.randint(*batch_size_range)
    token_number = random.randint(*token_number_range)
    vocab_size = random.randint(*vocab_size_range)
    hidden_size = random.randint(*hidden_size_range)

    ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, token_number)
    )

    return (vocab_size, hidden_size), ids


def create_model(
    hidden_size,
    vocab_size
):
    model = TestModel(
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    return model


@pytest.fixture
def input_data():
    return create_ids_example()


def test_scalar_sgdw(input_data):
    (vocab_size, hidden_size), ids = input_data
    model = create_model(hidden_size, vocab_size)
    x = model(ids)
    w_test = collect_layer_norm_weights(model)
    w_true = [model.layer_norm_1.weight, model.layer_norm_2.weight]
    w_true = torch.cat(w_true)
    w_test = torch.cat(w_test)
    ans = w_test == w_true
    assert ans
