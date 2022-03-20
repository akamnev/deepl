"""Shared Global Workspace layers"""
import torch
import torch.nn as nn
from .dropout import VariationalNormalEpanechnikovDropout
from .activations import get_activation
from ..models.config import GatingKind


def _build_position_index(
        batch_size, head_size, token_size, attention_half_width, device
):
    q_ids = torch.arange(token_size, dtype=torch.long, device=device)
    k_ids = torch.arange(token_size, dtype=torch.long, device=device)
    index = q_ids.view(-1, 1) - k_ids.view(1, -1)

    index = torch.clamp(-index + attention_half_width, 0, 2 * attention_half_width)
    index = index.unsqueeze(0).unsqueeze(0)
    index = index.expand([batch_size, head_size, token_size, token_size])
    return index


class LocalSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_half_width
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_half_width = attention_half_width

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_projection = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.value_projection = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.softmax = nn.Softmax(dim=-1)

        position_shape = (
            1,
            num_attention_heads,
            self.attention_head_size,
            2 * self.attention_half_width + 1
        )
        self.position_query = nn.Parameter(torch.Tensor(*position_shape))
        self.position_key = nn.Parameter(torch.Tensor(*position_shape))

        self.dropout = VariationalNormalEpanechnikovDropout(
            input_size=self.all_head_size
        )
        self.output_layer = nn.Linear(self.all_head_size, hidden_size)

        self._value_tensor = None
        self._score_tensor = None
        self._proba_tensor = None
        self._attention_mask_tensor = None

    def transpose_for_scores(self, x):
        new_x_shape = (
            x.size()[0], x.size()[1], self.num_attention_heads,
            self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_context(self, context):
        context = context.permute(0, 2, 1, 3)
        new_shape = context.size()[:-2] + (self.all_head_size, )
        context = context.reshape(*new_shape)
        return context

    def _multi_head_attention(self, query, key, value, mask, index=None):
        score = torch.matmul(query, key.transpose(-1, 2))
        extended_mask = torch.ones_like(score, dtype=torch.bool)
        if index is None:
            index = _build_position_index(
                query.shape[0],
                query.shape[1],
                query.shape[2],
                self.attention_half_width,
                query.device
            )

        # context -> position
        score_c2p = torch.matmul(query, self.position_key)
        score_c2p = torch.gather(score_c2p, dim=-1, index=index)
        # position -> context
        score_p2c = torch.matmul(key, self.position_query)
        score_p2c = torch.gather(score_p2c, dim=-1, index=index).transpose(-1, -2)
        score = score + score_c2p + score_p2c

        extended_mask *= mask.unsqueeze(1).unsqueeze(1)
        extended_mask *= mask.unsqueeze(-1).unsqueeze(1)

        ones = torch.ones_like(score, dtype=torch.bool)
        ma = torch.triu(ones, diagonal=-self.attention_half_width)
        mb = torch.triu(ones, diagonal=self.attention_half_width + 1)
        extended_mask *= ma
        extended_mask *= ~ mb

        lv = torch.finfo(score.dtype).min
        score = score + lv * (~extended_mask)

        proba = self.softmax(score)
        context = torch.matmul(proba, value)

        if self.training:
            self._score_tensor = score
            self._proba_tensor = proba

        return context, proba

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_index=None
    ):
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(dtype=torch.bool)

        if self.training:
            self._attention_mask_tensor = attention_mask

        query = self.transpose_for_scores(self.query_projection(hidden_states))
        key = self.transpose_for_scores(self.key_projection(hidden_states))
        value = self.transpose_for_scores(self.value_projection(hidden_states))
        context, proba = self._multi_head_attention(
            query, key, value, attention_mask, index=position_index
        )
        context = self.transpose_context(context)
        context = self.dropout(context, attention_mask)
        output = self.output_layer(context)

        if self.training:
            self._value_tensor = value
        return output, proba

    def loss_value_unity(self):
        mask = self._attention_mask_tensor
        value = self._value_tensor
        reg = torch.norm(value, dim=-1)
        loss = (1.0 - reg) ** 2
        loss = loss * mask[:, None, :]
        norm = value.shape[1] * torch.sum(mask)
        loss = torch.sum(loss)
        loss = loss / norm
        return loss

    def loss_attention_entropy(self):
        s = self._score_tensor - torch.logsumexp(self._score_tensor, dim=-1, keepdim=True)
        p = self._proba_tensor
        mask = self._attention_mask_tensor
        mask = mask[:, None, :, None] * mask[:, None, None, :]
        loss = p * s * mask
        norm = torch.sum(self._attention_mask_tensor) * self.num_attention_heads
        loss = torch.sum(loss) / norm
        return loss


class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            hidden_act
    ):
        super().__init__()
        self.dense_input = nn.Linear(hidden_size, intermediate_size)
        self.dropout = VariationalNormalEpanechnikovDropout(
            input_size=intermediate_size
        )
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dense_output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask):
        intermediate_states = self.dense_input(hidden_states)
        intermediate_states = self.dropout(intermediate_states, attention_mask)
        intermediate_states = self.intermediate_act_fn(intermediate_states)
        output_states = self.dense_output(intermediate_states)
        return output_states


class NoGating(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            update,
            hidden,
            eps=1e-6
    ):
        reg = None
        if self.training:
            h = torch.norm(hidden, dim=-1)
            u = torch.norm(update, dim=-1)
            reg = (None, (h - u) / (torch.abs(h + u) + eps))

        return update + hidden, reg


class VectorGating(nn.Module):
    def __init__(
            self,
            hidden_size
    ):
        super().__init__()
        self.wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.uh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
            self,
            update,
            hidden
    ):
        f = torch.sigmoid(self.wh(hidden) + self.uh(update))
        hidden = hidden + f * update

        reg = None
        if self.training:
            reg = self.reg_sigma_argument(h=hidden, u=update)
        return hidden, reg

    def reg_sigma_argument(self, h, u, eps=1e-6):
        """ Регуляризация:
            1. Аргумент сигма функции должен быть порядка единице по модулю.
            2. Модули скрытого состояний и обновления не должны сильно
               различаться

        Args:
            h: вектор скрытых состояний
            u: вектор внешней информации
            eps: (float) регуляризация
        """
        wh = self.wh.weight
        uh = self.uh.weight
        b = self.uh.bias

        h = torch.norm(h, dim=-1, keepdim=True)
        u = torch.norm(u, dim=-1, keepdim=True)
        wh = torch.norm(wh, dim=-1)
        uh = torch.norm(uh, dim=-1)

        reg_1 = wh * h + uh * u + torch.norm(b)
        reg_2 = ((h - u) / (torch.abs(h + u) + eps)).squeeze()
        return reg_1, reg_2


class ScalaGating(nn.Module):
    def __init__(
            self,
            hidden_size
    ):
        super().__init__()
        self.wh = nn.Linear(hidden_size, 1, bias=False)
        self.uh = nn.Linear(hidden_size, 1, bias=True)

    def forward(
            self,
            update,
            hidden
    ):
        f = torch.sigmoid(self.wh(hidden) + self.uh(update))
        hidden = hidden + f * update

        reg = None
        if self.training:
            reg = self.reg_sigma_argument(h=hidden, u=update)
        return hidden, reg

    def reg_sigma_argument(self, h, u, eps=1e-6):
        """ Регуляризация:
            1. Аргумент сигма функции должен быть порядка единице по модулю.
            2. Модули скрытого состояний и обновления не должны сильно
               различаться

        Args:
            h: вектор скрытых состояний
            u: вектор внешней информации
            eps: (float) регуляризация
        """
        wh = self.wh.weight
        uh = self.uh.weight
        b = self.uh.bias

        h = torch.norm(h, dim=-1)
        u = torch.norm(u, dim=-1)
        wh = torch.norm(wh)
        uh = torch.norm(uh)
        reg_1 = wh * h + uh * u + torch.abs(b)
        reg_2 = (h - u) / (torch.abs(h + u) + eps)
        return reg_1, reg_2


class SharedWorkSpace(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            gating,
            max_position=None,
            layer_norm_eps=1e-8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_h2m = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.key_h2m = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.value_h2m = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.dropout_h2m = VariationalNormalEpanechnikovDropout(
            input_size=self.all_head_size
        )
        self.output_layer_h2m = nn.Linear(self.all_head_size, hidden_size)

        self.query_m2h = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.key_m2h = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.value_m2h = nn.Linear(
            hidden_size, self.all_head_size, bias=False
        )
        self.dropout_m2h = VariationalNormalEpanechnikovDropout(
            input_size=self.all_head_size
        )
        self.output_layer_m2h = nn.Linear(self.all_head_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)

        if gating == GatingKind.NONE:
            self.gating = NoGating()
        elif gating == GatingKind.VectorGating:
            self.gating = VectorGating(hidden_size=hidden_size)
        elif gating == GatingKind.ScalaGating:
            self.gating = ScalaGating(hidden_size=hidden_size)
        else:
            raise ValueError(
                f'An unknown value `{gating}` of gating is specified '
            )
        self.position_key = None
        if max_position is not None:
            position_shape = (
                1,
                num_attention_heads,
                max_position,
                self.attention_head_size
            )
            self.position_key = nn.Parameter(torch.Tensor(*position_shape))
        self.layer_norm_workspace = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps
        )

    def transpose_for_scores(self, x):
        new_x_shape = (
            x.size()[0], x.size()[1], self.num_attention_heads,
            self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            workspace_states,
            hidden_states,
            attention_mask
    ):
        # update memory
        query_ws = self.transpose_for_scores(self.query_h2m(workspace_states))
        key_hs = self.transpose_for_scores(self.key_h2m(hidden_states))
        value_hs = self.transpose_for_scores(self.value_h2m(hidden_states))

        if self.position_key is not None:
            t = key_hs.shape[-2]
            key_hs = key_hs + self.position_key[..., :t, :]

        context_ws, proba_h2m = self._multi_head_attention(
            query_ws, key_hs, value_hs, attention_mask
        )
        context_ws = self.transpose_context(context_ws)
        context_ws = self.dropout_h2m(context_ws)
        output_ws = self.output_layer_h2m(context_ws)

        # gating
        workspace_states, reg_gating = self.gating(
            update=output_ws,
            hidden=workspace_states
        )
        workspace_states = self.layer_norm_workspace(workspace_states)
        # update hidden states
        query_hs = self.transpose_for_scores(self.query_m2h(hidden_states))
        key_ws = self.transpose_for_scores(self.key_m2h(workspace_states))
        value_ws = self.transpose_for_scores(self.value_m2h(workspace_states))

        context_hs, proba_m2h = self._multi_head_attention(
            query_hs, key_ws, value_ws
        )
        context_hs = self.transpose_context(context_hs)
        context_hs = self.dropout_m2h(context_hs, attention_mask)
        output_hs = self.output_layer_m2h(context_hs)

        return workspace_states, output_hs, (proba_h2m, proba_m2h), reg_gating

    def transpose_context(self, context):
        context = context.permute(0, 2, 1, 3)
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.reshape(*new_shape)
        return context

    def _multi_head_attention(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(-1, 2))
        if mask is not None:
            extended_mask = mask.unsqueeze(1).unsqueeze(1)
            lv = torch.finfo(score.dtype).min
            score = score + lv * (~extended_mask)

        proba = self.softmax(score)
        context = torch.matmul(proba, value)

        return context, proba


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            attention_half_width,
            hidden_act,
            shared_work_space_unit,
            layer_norm_eps=1e-8
    ):
        super().__init__()
        self.layer_norm_local_self_attention = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps
        )
        self.layer_norm_global_attention = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps
        )
        self.layer_norm_feedforward = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps
        )
        self.local_self_attention = LocalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_half_width=attention_half_width
        )
        self.shared_work_space_unit = shared_work_space_unit

        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act)

    def forward(
        self,
        workspace_states,
        hidden_states,
        attention_mask,
        position_index=None
    ):
        attention_output, proba_lsa = self.local_self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_index=position_index
        )
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm_local_self_attention(hidden_states)

        shared_work_space_output = self.shared_work_space_unit(
            workspace_states=workspace_states,
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        workspace_states = shared_work_space_output[0]
        # TODO: add gating mechanism
        hidden_states = hidden_states + shared_work_space_output[1]
        proba_ws_h2m, proba_ws_m2h = shared_work_space_output[2]
        reg_gating = shared_work_space_output[3]

        hidden_states = self.layer_norm_global_attention(hidden_states)

        feedforward_output = self.feedforward(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + feedforward_output

        hidden_states = self.layer_norm_feedforward(hidden_states)

        regularisation = reg_gating
        return workspace_states, hidden_states, \
               (proba_lsa, proba_ws_h2m, proba_ws_m2h), regularisation


class Encoder(nn.Module):
    def __init__(
            self,
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            intermediate_size,
            attention_half_width,
            hidden_act='ReLU',
            gating=GatingKind.NONE,
            max_position=None,
            layer_norm_eps=1e-8
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        shared_work_space_unit = SharedWorkSpace(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            gating=gating,
            max_position=max_position
        )
        self.layer = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_half_width=attention_half_width,
                hidden_act=hidden_act,
                shared_work_space_unit=shared_work_space_unit,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_hidden_layers)])

    def forward(
        self,
        workspace_states,
        hidden_states,
        attention_mask,
        n_layer=None,
        output_hidden_states=False,
        output_proba=False,
        output_regularisation=False
    ):
        all_hidden_states = []
        all_workspace_states = []
        all_proba_lsa = []
        all_proba_ws_h2m = []
        all_proba_ws_m2h = []
        all_reg_h2m_sigma_arg = []
        all_reg_h2m_diff_norm = []
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(dtype=torch.bool)

        for n, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_workspace_states.append(workspace_states)
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(
                workspace_states,
                hidden_states,
                attention_mask
            )
            workspace_states = layer_outputs[0]
            hidden_states = layer_outputs[1]
            if output_proba:
                all_proba_lsa.append(layer_outputs[2][0])
                all_proba_ws_h2m.append(layer_outputs[2][1])
                all_proba_ws_m2h.append(layer_outputs[2][2])
            if output_regularisation:
                all_reg_h2m_sigma_arg.append(layer_outputs[3][0])
                all_reg_h2m_diff_norm.append(layer_outputs[3][1])

            if n_layer is not None and n + 1 >= n_layer:
                break

        # Add last layer
        if output_hidden_states:
            all_workspace_states.append(workspace_states)
            all_hidden_states.append(hidden_states)

        outputs = [
            workspace_states, hidden_states, None, None, None, None, None,
            None, None
        ]

        if output_hidden_states:
            outputs[2] = all_workspace_states
            outputs[3] = all_hidden_states
        if output_proba:
            outputs[4] = all_proba_lsa
            outputs[5] = all_proba_ws_h2m
            outputs[6] = all_proba_ws_m2h
        if output_regularisation:
            outputs[7] = all_reg_h2m_sigma_arg
            outputs[8] = all_reg_h2m_diff_norm

        return outputs


class Embeddings(nn.Module):
    def __init__(
            self,
            workspace_size,
            vocab_size,
            hidden_size
    ):
        super().__init__()
        self.init_workspace = nn.Parameter(torch.Tensor(
            1, workspace_size, hidden_size
        ))

        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.dropout_ws = VariationalNormalEpanechnikovDropout(hidden_size)
        self.dropout_emb = VariationalNormalEpanechnikovDropout(hidden_size)

    def forward(self, input_ids, attention_mask):
        bs = input_ids.shape[0]
        workspace = self.init_workspace.repeat([bs, 1, 1])
        embeddings = self.word_embeddings(input_ids)

        workspace = self.dropout_ws(workspace)
        embeddings = self.dropout_emb(embeddings, attention_mask)

        return workspace, embeddings
