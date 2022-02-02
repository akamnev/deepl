"""Shared Global Workspace layers"""
import torch

from .utils import get_min_value
from ..models.config import AttentionType
from .encoders import BertSelfAttention


class LocalSelfAttention(BertSelfAttention):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            half_width_key=5,
            half_width_val=5,
            dropout_alpha=0.0,
            attention_head_size=None,
            attention_half_width=5,
            attention_type=AttentionType.BIDIRECTIONAL,
            self_attention_bias=False
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            half_width_key=half_width_key,
            half_width_val=half_width_val,
            dropout_alpha=dropout_alpha,
            attention_head_size=attention_head_size,
            attention_half_width=attention_half_width,
            attention_type=attention_type,
            self_attention_bias=self_attention_bias
        )
        assert attention_half_width is not None, \
            '`attention_half_width` must be int'
        assert attention_half_width > 0, \
            '`attention_half_width` must be greater than zero'
        assert half_width_key == 0 or half_width_key == attention_half_width, \
            'the value `half_width_key` must be equal `attention_half_width`'

    def get_attention_probs(self, query_layer, key_layer, attention_mask):
        attention_scores, extended_attention_mask = self._local_score_matrix(
            query_layer, key_layer, attention_mask
        )

        if self.half_width_key > 0:
            attention_scores_pos = self.relative_pos_key(query_layer)
            attention_scores = attention_scores + attention_scores_pos

        if self.attention_type == AttentionType.AUTOREGRESSION:
            auto_regression_mask = torch.ones_like(attention_scores, dtype=torch.bool)
            auto_regression_mask[..., :, self.attention_half_width+1:] = False
            extended_attention_mask *= auto_regression_mask
        if self.training:
            self._extended_attention_mask_tensor = extended_attention_mask
        extended_attention_mask = (~extended_attention_mask).to(torch.float32)
        extended_attention_mask *= get_min_value(extended_attention_mask)
        attention_scores = attention_scores + extended_attention_mask

        if self.training:
            self._score_tensor = attention_scores

        attention_scores = self.dropout_attention_scores(attention_scores)
        attention_probs = self.softmax(attention_scores)
        return attention_probs

    def _local_score_matrix(self, query_layer, key_layer, attention_mask):
        n = query_layer.shape[-2]  # allowed only for square matrix
        device = query_layer.device
        key_layer = key_layer.transpose(-1, -2)

        w = 2 * self.attention_half_width + 1
        key_layer_reordered = torch.zeros(
            key_layer.shape[:-1] + (n * w, ), device=device
        )
        extended_attention_mask = torch.zeros(
            key_layer.shape[:-2] + (n * w, ), device=device, dtype=torch.bool
        )

        ids_a_k = self._idx_val(n, n, self.attention_half_width, device)
        ids_v_k = self._idx_key(n, n, self.attention_half_width, device)

        key_layer_reordered[..., ids_a_k] = key_layer[..., ids_v_k]
        key_layer_reordered = key_layer_reordered.view(key_layer_reordered.shape[:-1] + (n, w))
        key_layer_reordered = key_layer_reordered.transpose(-2, -3)
        attention_scores = torch.sum(
            query_layer[..., None] * key_layer_reordered, dim=-2
        )

        if attention_mask is not None:
            extended_attention_mask[..., ids_a_k] = attention_mask[:, None, ids_v_k].to(torch.bool)
        else:
            extended_attention_mask[..., ids_a_k] = True
        extended_attention_mask = extended_attention_mask.view(extended_attention_mask.shape[:-1] + (n, w))

        return attention_scores, extended_attention_mask

    @staticmethod
    def _idx_key(n: int, m: int, half_width: int, device: torch.device):
        ii = torch.arange(0, n, device=device)
        jj = torch.arange(-half_width, half_width + 1, dtype=torch.int64, device=device)
        jj = jj[None, :]
        ii = ii[:, None]
        mask = ii + jj
        mask = (mask >= 0) * (mask < m)
        ids_att = ii + jj
        ids_att = ids_att[mask]
        return ids_att

    def get_context_layer(self, attention_probs, value_layer):
        context_layer = self._local_context_layer(attention_probs, value_layer)

        if self.half_width_val > 0:
            context_layer_pos = self.relative_pos_val(attention_probs)
            context_layer = context_layer + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def _local_context_layer(self, attention_probs, value_layer):
        n = attention_probs.shape[-2]
        m = value_layer.shape[-2]
        device = value_layer.device
        value_layer = value_layer.transpose(-1, -2)

        w = 2 * self.attention_half_width + 1
        value_layer_reordered = torch.zeros(
            value_layer.shape[:-1] + (n * w, ), device=device
        )

        ids_a_k = self._idx_val(n, m, self.attention_half_width, device)
        ids_att = self._idx_key(n, m, self.attention_half_width, device)

        value_layer_reordered[..., ids_a_k] = value_layer[..., ids_att]
        value_layer_reordered = value_layer_reordered.view(value_layer_reordered.shape[:-1] + (n, w))
        value_layer_reordered = value_layer_reordered.permute(0, 1, 3, 4, 2)
        context_layer = torch.sum(
            attention_probs[..., None] * value_layer_reordered, dim=-2
        )
        return context_layer

