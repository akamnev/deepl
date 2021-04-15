import math
import torch
import torch.nn as nn

from .activations import gelu, ACT2FN
from .utils import get_min_value
from ..models.config import PSS


class BertSelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_head=0.0,
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
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_prob = dropout_prob
        self.dropout_head = dropout_head

        self.half_width_key = half_width_key
        if half_width_key > 0:
            self.relative_pos_key = nn.Linear(self.attention_head_size,
                                              2 * self.half_width_key + 1,
                                              bias=False)
        self.half_width_val = half_width_val
        if half_width_val > 0:
            self.relative_pos_val = nn.Linear(2 * self.half_width_val + 1,
                                              self.attention_head_size,
                                              bias=False)

    def transpose_for_scores(self, x):
        new_x_shape = (x.size()[0], x.size()[1],
                       self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def dropout_attention_scores(self, scores):
        if self.training:
            mask = torch.ones(scores.shape, dtype=scores.dtype,
                              device=scores.device) * self.dropout_prob
            mask = torch.bernoulli(mask)
            scores = scores + mask * get_min_value(scores)
        return scores

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        query_layer, key_layer, value_layer, attention_mask = self.get_query_key_value(
            hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
        attention_probs = self.get_attention_probs(query_layer, key_layer, attention_mask)
        attention_probs = self.mask_headers(attention_probs, head_mask)
        context_layer = self.get_context_layer(attention_probs, value_layer)
        outputs = [context_layer, attention_probs] if self.output_attentions \
            else [context_layer]
        return outputs

    def get_query_key_value(self,
                            hidden_states,
                            attention_mask,
                            encoder_hidden_states,
                            encoder_attention_mask):
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        return query_layer, key_layer, value_layer, attention_mask

    def get_attention_probs(self, query_layer, key_layer, attention_mask):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.half_width_key > 0:
            attention_scores_pos = self.get_key_position_score(query_layer, key_layer)
            attention_scores = attention_scores + attention_scores_pos
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            extended_attention_mask = 1.0 - attention_mask[:, None, None, :]
            extended_attention_mask *= get_min_value(extended_attention_mask)
            attention_scores = attention_scores + extended_attention_mask

        attention_scores = self.dropout_attention_scores(attention_scores)
        attention_probs = self.softmax(attention_scores)
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

    def mask_headers(self, attention_probs, mask):
        # TODO: create head-dropout here
        if mask is not None:
            attention_probs = attention_probs * mask
        return attention_probs

    def get_key_position_score(self, query_layer, key_layer):
        n = query_layer.shape[-2]
        m = key_layer.shape[-2]
        a_k = self.relative_pos_key(query_layer)
        attention_scores_pos = torch.zeros(query_layer.shape[:2] + (n * m, ),
                                           device=query_layer.device)
        ids_att = [i * m + j
                   for i in range(n)
                   for j in range(max(0, i-self.half_width_key), min(m, i + self.half_width_key + 1))]
        ids_a_k = [i * (2 * self.half_width_key + 1) + j
                   for i in range(n)
                   for j in range(2 * self.half_width_key + 1)
                   if self.half_width_key <= i + j < m + self.half_width_key]
        a_k = a_k.view(a_k.shape[:-2] + (-1, ))[:, :, ids_a_k]
        attention_scores_pos[:, :, ids_att] = a_k
        attention_scores_pos = attention_scores_pos.view(attention_scores_pos.shape[:-1] + (n, m))
        return attention_scores_pos

    def get_val_position_score(self, attention_probs):
        n = attention_probs.shape[-2]
        m = attention_probs.shape[-1]
        w = 2 * self.half_width_val + 1
        attention_scores_pos = torch.zeros(
            attention_probs.shape[:2] + (n * w, ),
            device=attention_probs.device)
        ids_att = [i * m + j
                   for i in range(n)
                   for j in range(max(0, i-self.half_width_val), min(m, i + self.half_width_val + 1))]
        ids_a_v = [i * (2 * self.half_width_val + 1) + j
                   for i in range(n)
                   for j in range(2 * self.half_width_val + 1)
                   if self.half_width_val <= i + j < m + self.half_width_val]
        attention_scores_pos[:, :, ids_a_v] = attention_probs.view(attention_probs.shape[:-2] + (-1,))[:, :, ids_att]
        attention_scores_pos = attention_scores_pos.view(attention_scores_pos.shape[:-1] + (n, w))
        attention_scores_pos = self.relative_pos_val(attention_scores_pos)
        return attention_scores_pos


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      half_width_key=half_width_key,
                                      half_width_val=half_width_val,
                                      dropout_head=dropout_head,
                                      dropout_prob=dropout_prob,
                                      output_attentions=output_attentions)
        self.output = BertSelfOutput(hidden_size=hidden_size,
                                     dropout_prob=dropout_prob,
                                     layer_norm_eps=layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        self_outputs = self.self(hidden_states,
                                 attention_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = [attention_output] + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size,
                 hidden_act='gelu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size,
                 dropout_prob=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 half_width_key=0,
                 half_width_val=0,
                 is_decoder=False,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.attention = BertAttention(hidden_size=hidden_size,
                                       num_attention_heads=num_attention_heads,
                                       half_width_key=half_width_key,
                                       half_width_val=half_width_val,
                                       dropout_head=dropout_head,
                                       dropout_prob=dropout_prob,
                                       layer_norm_eps=layer_norm_eps,
                                       output_attentions=output_attentions)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.cross_attention = BertAttention(hidden_size=hidden_size,
                                                 num_attention_heads=num_attention_heads,
                                                 half_width_key=half_width_key,
                                                 half_width_val=half_width_val,
                                                 dropout_head=dropout_head,
                                                 dropout_prob=dropout_prob,
                                                 layer_norm_eps=layer_norm_eps,
                                                 output_attentions=output_attentions)
        self.intermediate = BertIntermediate(hidden_size=hidden_size,
                                             intermediate_size=intermediate_size,
                                             hidden_act=hidden_act)
        self.output = BertOutput(hidden_size=hidden_size,
                                 intermediate_size=intermediate_size,
                                 dropout_prob=dropout_prob,
                                 layer_norm_eps=layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(attention_output,
                                                           attention_mask,
                                                           encoder_hidden_states,
                                                           encoder_attention_mask
                                                           )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = [layer_output] + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,
                 half_width_key=0,
                 half_width_val=0,
                 is_decoder=False,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12,
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        if cross_layer_parameter_sharing == PSS.NO_PARAMETERS_SHARING:
            self.layer = nn.ModuleList([BertLayer(hidden_size=hidden_size,
                                                  num_attention_heads=num_attention_heads,
                                                  intermediate_size=intermediate_size,
                                                  half_width_key=half_width_key,
                                                  half_width_val=half_width_val,
                                                  is_decoder=is_decoder,
                                                  dropout_head=dropout_head,
                                                  dropout_prob=dropout_prob,
                                                  hidden_act=hidden_act,
                                                  layer_norm_eps=layer_norm_eps,
                                                  output_attentions=output_attentions)
                                        for _ in range(num_hidden_layers)])
        elif cross_layer_parameter_sharing == PSS.ALL_PARAMETERS_SHARING:
            self.single_layer = BertLayer(hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=intermediate_size,
                                          half_width_key=half_width_key,
                                          half_width_val=half_width_val,
                                          is_decoder=is_decoder,
                                          dropout_head=dropout_head,
                                          dropout_prob=dropout_prob,
                                          hidden_act=hidden_act,
                                          layer_norm_eps=layer_norm_eps,
                                          output_attentions=output_attentions)
        else:
            raise ValueError(f'{cross_layer_parameter_sharing} not recognized.'
                             f' `cross_layer_parameter_sharing` '
                             f' should be set to either '
                             f'`PSS.NO_PARAMETERS_SHARING`, '
                             f'`PSS.ALL_PARAMETERS_SHARING`.')
        self.num_hidden_layers = num_hidden_layers
        self.cross_layer_parameter_sharing = cross_layer_parameter_sharing

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        all_hidden_states = []
        all_attentions = []

        if self.cross_layer_parameter_sharing == PSS.NO_PARAMETERS_SHARING:
            layer = self.layer
        elif self.cross_layer_parameter_sharing == PSS.ALL_PARAMETERS_SHARING:
            layer = [self.single_layer for _ in range(self.num_hidden_layers)]
        else:
            raise ValueError(f'{self.cross_layer_parameter_sharing} not '
                             f'recognized. `cross_layer_parameter_sharing` '
                             f' should be set to either `None`,'
                             f' `all_parameters_sharing`.')

        for layer_module in layer:

            if self.output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(hidden_states,
                                         attention_mask,
                                         encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions.extend(layer_outputs[1:])

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = [hidden_states]
        if self.output_hidden_states:
            all_hidden_states = [tensor.unsqueeze(dim=0)
                                 for tensor in all_hidden_states]
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
            outputs.append(all_hidden_states)
        if self.output_attentions:
            all_attentions = [tensor.unsqueeze(dim=0)
                              for tensor in all_attentions]
            all_attentions = torch.cat(all_attentions, dim=0)
            outputs.append(all_attentions)
        return outputs


class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.decoder(x)
        return x


class LMHeadCut(LMHead):

    def __init__(self, hidden_size, vocab_size, ignore_index):
        super().__init__(hidden_size, vocab_size)
        self.ignore_index = ignore_index
        self.hidden_size = hidden_size

    def forward(self, features, labels=None):
        if labels is not None:
            labels = labels.view(-1)
            features = features.view(-1, self.hidden_size)
            mask = labels != self.ignore_index
            labels = labels[mask]
            features = features[mask]
        x = super().forward(features)
        return x, labels


class GraphAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      half_width_key=half_width_key,
                                      half_width_val=half_width_val,
                                      dropout_head=dropout_head,
                                      dropout_prob=dropout_prob,
                                      output_attentions=output_attentions)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        self_outputs = self.self(hidden_states,
                                 attention_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask)
        attention_output = self.dense(self_outputs[0])
        attention_output = attention_output + hidden_states
        outputs = [attention_output] + self_outputs[1:]
        return outputs


class GraphConvEncoder(nn.Module):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 cross_layer_parameter_sharing=PSS.NO_PARAMETERS_SHARING,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        if cross_layer_parameter_sharing == PSS.NO_PARAMETERS_SHARING:
            self.layer = nn.ModuleList([GraphAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                half_width_key=half_width_key,
                half_width_val=half_width_val,
                dropout_head=dropout_head,
                dropout_prob=dropout_prob,
                layer_norm_eps=layer_norm_eps,
                output_attentions=output_attentions)
                    for _ in range(num_hidden_layers)])
        elif cross_layer_parameter_sharing == PSS.ALL_PARAMETERS_SHARING:
            self.single_layer = GraphAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                half_width_key=half_width_key,
                half_width_val=half_width_val,
                dropout_head=dropout_head,
                dropout_prob=dropout_prob,
                layer_norm_eps=layer_norm_eps,
                output_attentions=output_attentions)
        else:
            raise ValueError(f'{cross_layer_parameter_sharing} not recognized.'
                             f' `cross_layer_parameter_sharing` '
                             f' should be set to either '
                             f'`PSS.NO_PARAMETERS_SHARING`, '
                             f'`PSS.ALL_PARAMETERS_SHARING`.')
        self.num_hidden_layers = num_hidden_layers
        self.cross_layer_parameter_sharing = cross_layer_parameter_sharing

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        all_hidden_states = []
        all_attentions = []

        if self.cross_layer_parameter_sharing == PSS.NO_PARAMETERS_SHARING:
            layer = self.layer
        elif self.cross_layer_parameter_sharing == PSS.ALL_PARAMETERS_SHARING:
            layer = [self.single_layer for _ in range(self.num_hidden_layers)]
        else:
            raise ValueError(f'{self.cross_layer_parameter_sharing} not '
                             f'recognized. `cross_layer_parameter_sharing` '
                             f' should be set to either `None`,'
                             f' `all_parameters_sharing`.')

        for layer_module in layer:

            if self.output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(hidden_states,
                                         attention_mask,
                                         encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions.extend(layer_outputs[1:])

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = [hidden_states]
        if self.output_hidden_states:
            all_hidden_states = [tensor.unsqueeze(dim=0)
                                 for tensor in all_hidden_states]
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
            outputs.append(all_hidden_states)
        if self.output_attentions:
            all_attentions = [tensor.unsqueeze(dim=0)
                              for tensor in all_attentions]
            all_attentions = torch.cat(all_attentions, dim=0)
            outputs.append(all_attentions)
        return outputs
