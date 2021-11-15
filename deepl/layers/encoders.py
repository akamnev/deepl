import torch
import torch.nn as nn

from .dropout import VariationalNormalEpanechnikovDropout
from .activations import get_activation
from .utils import get_min_value
from ..models.config import AttentionType


class BertSelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_alpha=0.0,
                 attention_head_size=None,
                 attention_type=AttentionType.BIDIRECTIONAL,
                 output_attentions=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        if attention_head_size is None:
            self.attention_head_size = int(hidden_size / num_attention_heads)
        else:
            self.attention_head_size = attention_head_size
        self.attention_type = attention_type

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_alpha = dropout_alpha

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
        self._value_tensor = None
        self._score_tensor = None
        self._proba_tensor = None
        self._attention_mask_tensor = None
        self._encoder_attention_mask_tensor = None

    def transpose_for_scores(self, x):
        new_x_shape = (x.size()[0], x.size()[1],
                       self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def dropout_attention_scores(self, scores):
        if self.training and self.dropout_alpha > 0:
            importance = self.softmax(scores.detach())
            importance = torch.pow(importance, self.dropout_alpha)
            importance /= torch.sum(importance, dim=-1, keepdim=True)
            mask = torch.bernoulli(1.0 - importance)
            scores = scores + mask * get_min_value(scores)
        return scores

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if self.training:
            self._attention_mask_tensor = attention_mask
            self._encoder_attention_mask_tensor = encoder_attention_mask

        query_layer, key_layer, value_layer, attention_mask = self.get_query_key_value(
            hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
        attention_probs = self.get_attention_probs(query_layer, key_layer, attention_mask)
        context_layer = self.get_context_layer(attention_probs, value_layer)
        outputs = [context_layer, attention_probs] if self.output_attentions \
            else [context_layer]
        if self.training:
            self._value_tensor = value_layer
            self._proba_tensor = attention_probs
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
        if attention_mask is not None:
            extended_attention_mask = 1.0 - attention_mask[:, None, None, :]
            extended_attention_mask *= get_min_value(extended_attention_mask)
            attention_scores = attention_scores + extended_attention_mask
        if self.attention_type == AttentionType.AUTOREGRESSION:
            auto_regression_mask = torch.triu(torch.ones_like(attention_scores), diagonal=1)
            auto_regression_mask *= get_min_value(auto_regression_mask)
            attention_scores = attention_scores + auto_regression_mask

        if self.training:
            self._score_tensor = attention_scores

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

    @staticmethod
    def _idx_att(n: int, m: int, half_width: int, device: torch.device):
        ii = torch.arange(0, n, device=device)
        jj = torch.arange(-half_width, half_width + 1, dtype=torch.int64, device=device)
        jj = jj[None, :]
        ii = ii[:, None]
        ss = m * ii
        mask = ii + jj
        mask = (mask >= 0) * (mask < m)
        ids_att = ii + jj + ss
        ids_att = ids_att[mask]
        return ids_att

    @staticmethod
    def _idx_val(n: int, m: int, half_width: int, device: torch.device):
        ii = torch.arange(0, n, device=device)
        jj = torch.arange(2 * half_width + 1, dtype=torch.int64, device=device)
        jj = jj[None, :]
        ii = ii[:, None]
        ss = ii * (2 * half_width + 1)
        mask = ii + jj
        mask = (half_width <= mask) * (mask < m + half_width)
        ids_a_k = jj + ss
        ids_a_k = ids_a_k[mask]
        return ids_a_k

    def get_key_position_score(self, query_layer, key_layer):
        n = query_layer.shape[-2]
        m = key_layer.shape[-2]
        device = query_layer.device
        a_k = self.relative_pos_key(query_layer)
        attention_scores_pos = torch.zeros(query_layer.shape[:2] + (n * m, ),
                                           device=query_layer.device)
        ids_a_k = self._idx_val(n, m, self.half_width_key, device)
        ids_att = self._idx_att(n, m, self.half_width_key, device)
        a_k = a_k.view(a_k.shape[:-2] + (-1, ))[..., ids_a_k]
        attention_scores_pos[..., ids_att] = a_k
        attention_scores_pos = attention_scores_pos.view(attention_scores_pos.shape[:-1] + (n, m))
        return attention_scores_pos

    def get_val_position_score(self, attention_probs):
        n = attention_probs.shape[-2]
        m = attention_probs.shape[-1]
        device = attention_probs.device
        w = 2 * self.half_width_val + 1
        attention_scores_pos = torch.zeros(
            attention_probs.shape[:2] + (n * w, ),
            device=attention_probs.device)
        ids_a_v = self._idx_val(n, m, self.half_width_val, device)
        ids_att = self._idx_att(n, m, self.half_width_val, device)
        attention_scores_pos[..., ids_a_v] = attention_probs.view(attention_probs.shape[:-2] + (-1,))[..., ids_att]
        attention_scores_pos = attention_scores_pos.view(attention_scores_pos.shape[:-1] + (n, w))
        attention_scores_pos = self.relative_pos_val(attention_scores_pos)
        return attention_scores_pos

    def loss_value_unity(self):
        if self._encoder_attention_mask_tensor is None:
            mask = self._attention_mask_tensor
        else:
            mask = self._encoder_attention_mask_tensor
        value = self._value_tensor
        value = value * mask[:, None, :, None]
        loss = (1.0 - torch.norm(value, dim=-1)) ** 2
        norm = value.shape[1] * torch.sum(mask)
        loss = torch.sum(loss)
        loss = loss / norm
        return loss

    def loss_attention_entropy(self):
        """ Идея аппроксимировать Maximal entropy random walk
        для матриц внимания. Для регуляризации необходимо максимизировать
        значение.
        """
        s = self._score_tensor - torch.logsumexp(self._score_tensor, dim=-1, keepdim=True)
        p = self._proba_tensor
        mask_output = self._attention_mask_tensor
        if self._encoder_attention_mask_tensor is not None:
            mask_input = self._encoder_attention_mask_tensor
        else:
            mask_input = self._attention_mask_tensor
        mask = mask_output[:, None, :, None] * mask_input[:, None, None, :]
        p = p * mask
        s = s * mask
        norm = torch.sum(mask[..., 0]) * self.num_attention_heads
        loss = torch.sum(p * s) / norm
        return loss

    def loss_score_mean(self):
        """ Идея зафиксировать среднее значение для score
        """
        s = self._score_tensor
        mask_output = self._attention_mask_tensor
        if self._encoder_attention_mask_tensor is not None:
            mask_input = self._encoder_attention_mask_tensor
        else:
            mask_input = self._attention_mask_tensor
        mask = mask_output[:, None, :, None] * mask_input[:, None, None, :]
        s = torch.sum(s * mask, dim=-1)
        norm = torch.sum(mask, dim=-1)
        s = s / (norm + 1e-8)
        loss = torch.mean(s ** 2)
        return loss


class BertAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 dropout_alpha=0.0,
                 attention_head_size=None,
                 attention_type=AttentionType.BIDIRECTIONAL,
                 output_attentions=False):
        super().__init__()
        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      half_width_key=half_width_key,
                                      half_width_val=half_width_val,
                                      dropout_alpha=dropout_alpha,
                                      attention_head_size=attention_head_size,
                                      attention_type=attention_type,
                                      output_attentions=output_attentions)
        self.dropout_self = VariationalNormalEpanechnikovDropout(
            input_size=self.self.all_head_size)
        self.dense_output = nn.Linear(self.self.all_head_size, hidden_size)
        self.dropout_output = VariationalNormalEpanechnikovDropout(
            input_size=hidden_size)

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
        self_attention_outputs = self.dropout_self(self_outputs[0], attention_mask)
        attention_outputs = self.dense_output(self_attention_outputs)
        attention_outputs = self.dropout_output(attention_outputs, attention_mask)
        outputs = [attention_outputs] + self_outputs[1:]
        return outputs


class BertFeedForward(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act):
        super().__init__()
        self.dense_input = nn.Linear(hidden_size, intermediate_size)
        self.dropout_input = VariationalNormalEpanechnikovDropout(
            input_size=intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.dropout_output = VariationalNormalEpanechnikovDropout(
            input_size=hidden_size)

    def forward(self, hidden_states, attention_mask):
        intermediate_states = self.dense_input(hidden_states)
        intermediate_states = self.dropout_input(intermediate_states, attention_mask)
        intermediate_states = self.intermediate_act_fn(intermediate_states)
        output_states = self.dense_output(intermediate_states)
        output_states = self.dropout_output(output_states, attention_mask)
        return output_states


class BertLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 half_width_key=0,
                 half_width_val=0,
                 is_decoder=False,
                 dropout_alpha=0.0,
                 attention_head_size=None,
                 hidden_act='ReLU',
                 attention_type=AttentionType.BIDIRECTIONAL,
                 layer_norm_eps=1e-8,
                 output_attentions=False):
        super().__init__()
        self.layer_norm_attention = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps)
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            half_width_key=half_width_key,
            half_width_val=half_width_val,
            dropout_alpha=dropout_alpha,
            attention_head_size=attention_head_size,
            attention_type=attention_type,
            output_attentions=output_attentions)

        self.is_decoder = is_decoder
        if self.is_decoder:
            self.layer_norm_cross_attention = nn.LayerNorm(
                hidden_size, eps=layer_norm_eps)
            self.cross_attention = BertAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                half_width_key=half_width_key,
                half_width_val=half_width_val,
                dropout_alpha=dropout_alpha,
                attention_head_size=attention_head_size,
                attention_type=AttentionType.BIDIRECTIONAL,
                output_attentions=output_attentions)

        self.feedforward = None
        if intermediate_size > 0:
            self.layer_norm_feedforward = nn.LayerNorm(
                hidden_size, eps=layer_norm_eps)
            self.feedforward = BertFeedForward(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):
        hidden_states = self.layer_norm_attention(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask)
        attention_output = self_attention_outputs[0]
        hidden_states = hidden_states + attention_output
        outputs = self_attention_outputs[1:]

        if self.is_decoder:
            hidden_states = self.layer_norm_cross_attention(hidden_states)
            cross_attention_outputs = self.cross_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            hidden_states = hidden_states + attention_output
            outputs = outputs + cross_attention_outputs[1:]

        if self.feedforward is not None:
            hidden_states = self.layer_norm_feedforward(hidden_states)
            feedforward_output = self.feedforward(
                hidden_states=hidden_states,
                attention_mask=attention_mask)
            hidden_states = hidden_states + feedforward_output

        outputs = [hidden_states] + outputs
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
                 dropout_alpha=0.0,
                 attention_head_size=None,
                 hidden_act='ReLU',
                 attention_type=AttentionType.BIDIRECTIONAL,
                 layer_norm_eps=1e-8,
                 output_attentions=False,
                 output_hidden_states=False):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.layer = nn.ModuleList([
            BertLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                half_width_key=half_width_key,
                half_width_val=half_width_val,
                is_decoder=is_decoder,
                dropout_alpha=dropout_alpha,
                attention_head_size=attention_head_size,
                hidden_act=hidden_act,
                attention_type=attention_type,
                layer_norm_eps=layer_norm_eps,
                output_attentions=output_attentions)
            for _ in range(num_hidden_layers)])
        self.num_hidden_layers = num_hidden_layers

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        n_layer=None,
    ):
        all_hidden_states = []
        all_self_attentions = []
        all_cross_attentions = []

        for n, layer_module in enumerate(self.layer):

            if self.output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(hidden_states,
                                         attention_mask,
                                         encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                if len(layer_outputs) > 1:
                    all_self_attentions.extend(layer_outputs[1])
                if len(layer_outputs) > 2:
                    all_cross_attentions.extend(layer_outputs[2])

            if n_layer is not None and n + 1 >= n_layer:
                break

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = [hidden_states, None, None, None]
        if all_self_attentions:
            all_self_attentions = [tensor.unsqueeze(dim=0)
                                   for tensor in all_self_attentions]
            all_self_attentions = torch.cat(all_self_attentions, dim=0)
            outputs[1] = all_self_attentions

        if self.output_hidden_states:
            all_hidden_states = [tensor.unsqueeze(dim=0)
                                 for tensor in all_hidden_states]
            all_hidden_states = torch.cat(all_hidden_states, dim=0)
            outputs[2] = all_hidden_states

        if all_cross_attentions:
            all_cross_attentions = [tensor.unsqueeze(dim=0)
                                    for tensor in all_cross_attentions]
            all_cross_attentions = torch.cat(all_cross_attentions, dim=0)
            outputs[3] = all_cross_attentions

        return outputs
