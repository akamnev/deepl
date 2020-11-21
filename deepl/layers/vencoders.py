import math
import torch
import torch.nn as nn

from .activations import gelu, ACT2FN
from .encoders import BertSelfAttention
from ..models.config import PSS


def kl_div(mu, sigma):
    """
    KL-divergence between a diagonal multivariate normal,
    and a standard normal distribution (with zero mean and unit variance)
    """
    sigma_2 = sigma * sigma
    kld = 0.5 * torch.mean(mu * mu + sigma_2 - torch.log(sigma_2) - 1.0)
    return kld


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.dense_mu = nn.Linear(hidden_size, hidden_size)
        self.dense_sigma = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        mu = self.dense_mu(hidden_states)
        sigma = self.dense_sigma(hidden_states)
        kld = kl_div(mu, sigma)
        z = mu + sigma * torch.randn(sigma.shape,
                                     dtype=sigma.dtype,
                                     device=sigma.device)
        return z, kld


class BertAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 half_width_key=0,
                 half_width_val=0,
                 temperature=1.0,
                 dropout_head=0.0,
                 dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 output_attentions=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      half_width_key=half_width_key,
                                      half_width_val=half_width_val,
                                      temperature=temperature,
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
        self_inputs = self.layer_norm(hidden_states)
        self_outputs = self.self(self_inputs,
                                 attention_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask)
        attention_outputs, kld = self.output(self_outputs[0])
        hidden_states = hidden_states + attention_outputs
        outputs = [hidden_states] + [kld] + self_outputs[1:]
        return outputs


class BertFeedForward(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense_mu = nn.Linear(hidden_size, intermediate_size)
        self.dense_sigma = nn.Linear(hidden_size, intermediate_size)
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        input_states = self.layer_norm(hidden_states)
        mu = self.dense_mu(input_states)
        mu = self.intermediate_act_fn(mu)
        sigma = self.dense_sigma(input_states)
        kld = kl_div(mu, sigma)
        intermediate_states = mu + sigma * torch.randn(sigma.shape,
                                                       dtype=sigma.dtype,
                                                       device=sigma.device)
        output_states = self.dense_output(intermediate_states)
        hidden_states = hidden_states + output_states
        return hidden_states, kld


class BertLayer(nn.Module):
    def __init__(self, hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 half_width_key=0,
                 half_width_val=0,
                 is_decoder=False,
                 temperature=1.0,
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
                                       temperature=temperature,
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
                                                 temperature=temperature,
                                                 dropout_head=dropout_head,
                                                 dropout_prob=dropout_prob,
                                                 layer_norm_eps=layer_norm_eps,
                                                 output_attentions=output_attentions)
        self.feedforward = BertFeedForward(hidden_size=hidden_size,
                                           intermediate_size=intermediate_size,
                                           hidden_act=hidden_act,
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
        attention_kld = [self_attention_outputs[1]]
        outputs = self_attention_outputs[2:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(attention_output,
                                                           attention_mask,
                                                           encoder_hidden_states,
                                                           encoder_attention_mask
                                                           )
            attention_output = cross_attention_outputs[0]
            attention_kld += [cross_attention_outputs[1]]
            outputs += cross_attention_outputs[2:]

        layer_output, ff_kld = self.feedforward(attention_output)
        bert_kld = attention_kld + [ff_kld]
        outputs = [layer_output] + [bert_kld] + [outputs]
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
                 temperature=1.0,
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
                                                  temperature=temperature,
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
                                          temperature=temperature,
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
        all_kld = []
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
            all_kld.extend(layer_outputs[1])

            if self.output_attentions:
                all_attentions.extend(layer_outputs[2])

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = [hidden_states, torch.stack(all_kld, dim=0)]
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
