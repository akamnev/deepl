import pytest
from tqdm import tqdm
import torch
import torch.nn as nn
import math
from deepl.layers.encoders import BertSelfAttention
from deepl.layers.utils import get_min_value

MAX_FLOAT32_ERROR = 1e-6


class BertSelfAttentionNaiveImpl(nn.Module):
    def __init__(self, hidden_size,
                 num_attention_heads,
                 half_width_key,
                 half_width_val,
                 dropout_prob=0.0,
                 output_attentions=False):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout_prob = dropout_prob

        # 0 for padding
        self.half_width_key = half_width_key
        self.relative_pos_key = nn.Embedding(2 * self.half_width_key + 1,
                                             self.attention_head_size)
        self.half_width_val = half_width_val
        self.relative_pos_val = nn.Embedding(2 * self.half_width_val + 1,
                                             self.attention_head_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        query_layer, key_layer, value_layer = self.get_query_key_value(hidden_states)
        attention_probs = self.get_attention_probs(query_layer, key_layer, attention_mask)
        context_layer = self.get_context_layer(attention_probs, value_layer)
        outputs = (context_layer, attention_probs) if self.output_attentions \
            else (context_layer,)
        return outputs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_query_key_value(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        return query_layer, key_layer, value_layer

    def get_attention_probs(self, query_layer, key_layer, attention_mask):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.half_width_key > 0:
            attention_scores_pos = self.get_key_position_score(query_layer)
            attention_scores = attention_scores + attention_scores_pos
        if attention_mask is not None:
            extended_attention_mask = 1.0 - attention_mask[:, None, None, :]
            extended_attention_mask *= get_min_value(extended_attention_mask)
            attention_scores = attention_scores + extended_attention_mask

        # attention_scores = self.dropout_attention_scores(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs

    def get_context_layer(self, attention_probs, value_layer):
        context_layer = torch.matmul(attention_probs, value_layer)
        if self.half_width_val > 0:
            context_layer_pos = self.get_val_position_score(attention_probs)
            context_layer = context_layer + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def get_key_position_score(self, query_layer):
        n = query_layer.shape[-2]
        attention_scores_pos = torch.zeros(query_layer.shape[:2] + (n, n),
                                           device=query_layer.device)
        for i in range(n):
            for j in range(n):
                pos_idx = j - i + self.half_width_key
                if 0 <= pos_idx < 2 * self.half_width_key + 1:
                    pos_idx = torch.tensor(pos_idx, device=query_layer.device,
                                           dtype=torch.long)
                    w = self.relative_pos_key(pos_idx).view(-1, 1)
                    q = torch.matmul(query_layer[:, :, i, :], w)
                    attention_scores_pos[:, :, i, j] = q.squeeze(-1)
        return attention_scores_pos

    def get_val_position_score(self, attention_probs):
        n = attention_probs.shape[-2]
        context_layer_pos = torch.zeros(attention_probs.shape[:2] + (n, self.attention_head_size),
                                        device=attention_probs.device)
        for i in range(n):
            for j in range(n):
                pos_idx = j - i + self.half_width_val
                if 0 <= pos_idx < 2 * self.half_width_val + 1:
                    pos_idx = torch.tensor(pos_idx, device=attention_probs.device,
                                           dtype=torch.long)
                    w = self.relative_pos_val(pos_idx).view(1, -1)
                    q = torch.matmul(attention_probs[:, :, i, j].unsqueeze(-1), w)
                    context_layer_pos[:, :, i, :] += q
        return context_layer_pos


def naive_test_obj():
    head_size = 8
    head_number = 2
    hidden_size = head_size * head_number
    half_width_key = 2
    half_width_value = 1
    params = {
        'hidden_size': hidden_size,
        'num_attention_heads': head_number,
        'half_width_key': half_width_key,
        'half_width_val': half_width_value,
        'output_attentions': False
    }
    obj_test = BertSelfAttention(**params)
    obj_naive = BertSelfAttentionNaiveImpl(**params)
    obj_naive.query = obj_test.query
    obj_naive.key = obj_test.key
    obj_naive.value = obj_test.value
    obj_naive.relative_pos_key.weight = nn.Parameter(obj_test.relative_pos_key.weight.detach())
    obj_naive.relative_pos_val.weight = nn.Parameter(obj_test.relative_pos_val.weight.detach().T)

    return obj_naive, obj_test, hidden_size


def get_input_data(hidden_size):
    batch_size = 2
    sequence_length = 5
    return torch.rand((batch_size, sequence_length, hidden_size))


def test_query_values():
    obj_naive, obj_test, hidden_size = naive_test_obj()
    input_data = get_input_data(hidden_size)
    q_naive, k_naive, v_naive = obj_naive.get_query_key_value(input_data)
    q_test, k_test, v_test, a_test = obj_test.get_query_key_value(input_data, None, None, None)
    dv = q_naive - q_test
    assert torch.max(torch.abs(dv)) < MAX_FLOAT32_ERROR


def test_position_key_score():
    obj_naive, obj_test, hidden_size = naive_test_obj()
    input_data = get_input_data(hidden_size)
    q_naive, k_naive, v_naive = obj_naive.get_query_key_value(input_data)
    q_test, k_test, v_test, a_test = obj_test.get_query_key_value(input_data, None, None, None)
    att_pos_naive = obj_naive.get_key_position_score(q_naive)
    att_pos_test = obj_test.get_key_position_score(q_test, k_test)
    dv = att_pos_naive - att_pos_test
    assert torch.max(torch.abs(dv)) < MAX_FLOAT32_ERROR


def test_attention_proba():
    obj_naive, obj_test, hidden_size = naive_test_obj()
    input_data = get_input_data(hidden_size)
    q_naive, k_naive, v_naive = obj_naive.get_query_key_value(input_data)
    q_test, k_test, v_test, a_test = obj_test.get_query_key_value(input_data, None, None, None)
    att_prob_naive = obj_naive.get_attention_probs(q_naive, k_naive, None)
    att_prob_test = obj_test.get_attention_probs(q_test, k_test, None)
    dv = att_prob_naive - att_prob_test
    assert torch.max(torch.abs(dv)) < MAX_FLOAT32_ERROR


def test_position_value_score():
    obj_naive, obj_test, hidden_size = naive_test_obj()
    input_data = get_input_data(hidden_size)
    q_naive, k_naive, v_naive = obj_naive.get_query_key_value(input_data)
    q_test, k_test, v_test, a_test = obj_test.get_query_key_value(input_data, None, None, None)
    att_prob_naive = obj_naive.get_attention_probs(q_naive, k_naive, None)
    att_prob_test = obj_test.get_attention_probs(q_test, k_test, None)

    att_pos_naive = obj_naive.get_val_position_score(att_prob_naive)
    att_pos_test = obj_test.get_val_position_score(att_prob_test)
    dv = att_pos_naive - att_pos_test
    assert torch.max(torch.abs(dv)) < MAX_FLOAT32_ERROR


if __name__ == '__main__':
    for _ in tqdm(range(100)):
        test_query_values()
        test_position_key_score()
        test_attention_proba()
        test_position_value_score()
