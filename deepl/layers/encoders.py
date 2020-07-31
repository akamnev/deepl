import math
import torch
import torch.nn as nn

from .activations import gelu, ACT2FN
from .utils import get_min_value, prune_linear_layer


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1,
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

    def transpose_for_scores(self, x):
        new_x_shape = (x.size()[0], x.size()[1], self.num_attention_heads,
                       self.attention_head_size)
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
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
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

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            extended_attention_mask = 1.0 - attention_mask[:, None, None, :]
            extended_attention_mask *= get_min_value(extended_attention_mask)

            attention_scores = attention_scores + extended_attention_mask

        attention_scores = self.dropout_attention_scores(attention_scores)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (context_layer.size()[0],
                                   context_layer.size()[1], self.all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = [context_layer, attention_probs] if self.output_attentions \
            else [context_layer]
        return outputs


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
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.self = BertSelfAttention(hidden_size,
                                      num_attention_heads,
                                      dropout_prob,
                                      output_attentions)
        self.output = BertSelfOutput(hidden_size,
                                     dropout_prob,
                                     layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads,
                          self.self.attention_head_size)
        # Convert to set and remove already pruned heads
        heads = set(heads) - self.pruned_heads
        for head in heads:
            # Compute how many pruned heads are before the head and move
            # the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        self_outputs = self.self(hidden_states,
                                 attention_mask,
                                 encoder_hidden_states,
                                 encoder_attention_mask)
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
                 is_decoder=False,
                 dropout_prob=0.1,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.attention = BertAttention(hidden_size,
                                       num_attention_heads,
                                       dropout_prob,
                                       layer_norm_eps,
                                       output_attentions)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.cross_attention = BertAttention(hidden_size,
                                                 num_attention_heads,
                                                 dropout_prob,
                                                 layer_norm_eps,
                                                 output_attentions)
        self.intermediate = BertIntermediate(hidden_size,
                                             intermediate_size,
                                             hidden_act)
        self.output = BertOutput(hidden_size, intermediate_size,
                                 dropout_prob, layer_norm_eps)

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
                 is_decoder=False,
                 dropout_prob=0.1,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12,
                 cross_layer_parameter_sharing=None,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        if cross_layer_parameter_sharing is None:
            self.layer = nn.ModuleList([BertLayer(hidden_size,
                                                  num_attention_heads,
                                                  intermediate_size,
                                                  is_decoder,
                                                  dropout_prob,
                                                  hidden_act,
                                                  layer_norm_eps,
                                                  output_attentions)
                                        for _ in range(num_hidden_layers)])
        elif cross_layer_parameter_sharing == 'all_parameters_sharing':
            self.single_layer = BertLayer(hidden_size,
                                          num_attention_heads,
                                          intermediate_size,
                                          is_decoder,
                                          dropout_prob,
                                          hidden_act,
                                          layer_norm_eps,
                                          output_attentions)
        else:
            raise ValueError(f'{cross_layer_parameter_sharing} not recognized.'
                             f' `cross_layer_parameter_sharing` '
                             f' should be set to either `None`,'
                             f' `all_parameters_sharing`.')
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

        if self.cross_layer_parameter_sharing is None:
            layer = self.layer
        else:
            layer = [self.single_layer for _ in range(self.num_hidden_layers)]

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
