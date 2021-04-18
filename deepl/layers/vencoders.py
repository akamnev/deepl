import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .activations import get_activation
from .encoders import BertSelfAttention
from ..models.config import PSS
from .utils import kld_gaussian


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, input_size=None, sigma_eps=1e-12):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.dense = nn.Linear(input_size, hidden_size)
        self.log_sigma = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.log_sigma.data.fill_(-1.0 - 0.5*math.log(input_size))
        self.sigma_eps = sigma_eps
        self.use_var = True

    def forward(self, hidden_states):
        mu = self.dense(hidden_states)
        if self.training and self.use_var:
            sigma_square = torch.exp(2.0 * self.log_sigma)
            variance = F.linear(hidden_states**2, sigma_square)
            variance = torch.sqrt(variance) + self.sigma_eps
            xi = torch.randn(variance.shape,
                             dtype=variance.dtype,
                             device=variance.device)
            xi = torch.fmod(xi, 2.5)
            mu = mu + variance * xi
        return mu

    def kld_gaussian(self, nu=0.0, rho=1.0):
        return kld_gaussian(self.dense.weight, self.log_sigma, nu=nu, rho=rho)


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
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      half_width_key=half_width_key,
                                      half_width_val=half_width_val,
                                      dropout_head=dropout_head,
                                      dropout_prob=dropout_prob,
                                      output_attentions=output_attentions)
        self.output = BertSelfOutput(hidden_size=hidden_size)

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
        attention_outputs = self.output(self_outputs[0])
        hidden_states = hidden_states + attention_outputs
        outputs = [hidden_states] + self_outputs[1:]
        return outputs


class BertFeedForward(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 layer_norm_eps=1e-12,
                 sigma_eps=1e-12):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense_input = nn.Linear(hidden_size, intermediate_size)
        self.log_sigma_input = nn.Parameter(torch.Tensor(intermediate_size, hidden_size))
        self.log_sigma_input.data.fill_(-1.0 - 0.5*math.log(hidden_size))
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.log_sigma_output = nn.Parameter(torch.Tensor(hidden_size, intermediate_size))
        self.log_sigma_output.data.fill_(-1.0 - 0.5*math.log(intermediate_size))
        self.intermediate_act_fn = get_activation(hidden_act)
        self.sigma_eps = sigma_eps
        self.use_var = True

    def forward(self, hidden_states):
        input_states = self.layer_norm(hidden_states)
        intermediate_states = self.dense_input(input_states)
        if self.training and self.use_var:
            sigma_square = torch.exp(2.0 * self.log_sigma_input)
            variance = F.linear(hidden_states**2, sigma_square)
            variance = torch.sqrt(variance) + self.sigma_eps
            xi = torch.randn(variance.shape,
                             dtype=variance.dtype,
                             device=variance.device)
            xi = torch.fmod(xi, 2.5)
            intermediate_states = intermediate_states + variance * xi
        intermediate_states = self.intermediate_act_fn(intermediate_states)
        output_states = self.dense_output(intermediate_states)
        if self.training and self.use_var:
            sigma_square = torch.exp(2.0 * self.log_sigma_output)
            variance = F.linear(intermediate_states**2, sigma_square)
            variance = torch.sqrt(variance) + self.sigma_eps
            xi = torch.randn(variance.shape,
                             dtype=variance.dtype,
                             device=variance.device)
            xi = torch.fmod(xi, 2.5)
            output_states = output_states + variance * xi
        hidden_states = hidden_states + output_states
        return hidden_states

    def kld_gaussian(self, nu=0.0, rho=1.0):
        kld_input = kld_gaussian(
            self.dense_input.weight, self.log_sigma_input, nu=nu, rho=rho)
        kld_output = kld_gaussian(
            self.dense_output.weight, self.log_sigma_output, nu=nu, rho=rho)
        return kld_input + kld_output


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
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(attention_output,
                                                           attention_mask,
                                                           encoder_hidden_states,
                                                           encoder_attention_mask
                                                           )
            attention_output = cross_attention_outputs[0]
            outputs += cross_attention_outputs[1:]

        layer_output, ff_kld = self.feedforward(attention_output)
        outputs = [layer_output] + [outputs]
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
                all_attentions.extend(layer_outputs[1])

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
