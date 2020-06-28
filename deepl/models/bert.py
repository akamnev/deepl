"""Bidirectional Encoder Representations from Transformers"""
import torch
from .base import BERTBase, LMMixin
from ..layers.embeddings import (AbsolutePositionEmbeddings,
                                 VectorTextFirstEmbeddings,
                                 VectorTextLastEmbeddings,
                                 VectorTextInsideEmbeddings)
from ..layers.encoders import BertEncoder, LMHead, LMHeadCut
from ..layers.utils import get_attention_mask
from .config import (BERTLanguageModelConfig,
                     VectorTextBERTConfig,
                     TextVectorVAEConfig)
from ..layers.vae import VAENormalTanhAbs

__all__ = ['BERT',
           'LanguageModel',
           'TextVectorMean',
           'TextVectorMax',
           'VectorText',
           'TextVectorMeanVAE']


class BERT(BERTBase):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = AbsolutePositionEmbeddings(config.vocab_size,
                                                    config.hidden_size,
                                                    config.max_position_embedding,
                                                    config.device,
                                                    config.padding_idx,
                                                    config.dropout_prob,
                                                    config.layer_norm_eps)
        self.encoder = BertEncoder(config.num_hidden_layers,
                                   config.num_attention_heads,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   config.is_decoder,
                                   config.dropout_prob,
                                   config.hidden_act,
                                   config.layer_norm_eps,
                                   config.cross_layer_parameter_sharing,
                                   config.output_attentions,
                                   config.output_hidden_states)

    def forward(self,
                input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        hidden_states = self.embedding(input_ids)

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        outputs = self.encoder(hidden_states,
                               attention_mask,
                               head_mask,
                               encoder_hidden_states,
                               encoder_attention_mask)
        return outputs


class LanguageModel(BERT, LMMixin):
    config_cls = BERTLanguageModelConfig

    def __init__(self, config):
        super().__init__(config)
        if config.use_cut_head:
            self.lm_head = LMHeadCut(config.hidden_size,
                                     config.vocab_size,
                                     config.ignore_index)
        else:
            self.lm_head = LMHead(config.hidden_size,
                                  config.vocab_size)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def init_weights(self):
        super().init_weights()

        if self.config.tie_embedding_vectors:
            self.tie_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                masked_lm_labels=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
        sequence_output = outputs[0]
        if masked_lm_labels is not None:
            max_length = max([len(x) for x in masked_lm_labels])
            masked_lm_labels = [x + [self.config.ignore_index] * (max_length - len(x))
                                for x in masked_lm_labels]
            masked_lm_labels = torch.tensor(masked_lm_labels,
                                            dtype=torch.long,
                                            device=self.config.device)

        if self.config.use_cut_head:
            prediction_scores, masked_lm_labels = self.lm_head(sequence_output,
                                                               masked_lm_labels)
        else:
            prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, ) + outputs[1:]

        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.config.ignore_index)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            outputs = (masked_lm_loss,) + outputs

        # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
        return outputs


class TextVectorMean(BERT):
    def forward(self,
                input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
        sequence_output = outputs[0]

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=sequence_output.dtype,
                                         device=self.config.device)

        norm = attention_mask.sum(axis=1, keepdim=True)
        norm[norm == 0.0] = 1.0
        attention_mask = attention_mask.unsqueeze(-1)
        vectors = (sequence_output * attention_mask).sum(axis=1) / norm
        outputs = (vectors, ) + outputs[1:]
        return outputs


class TextVectorMax(BERT):
    def forward(self,
                input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
        sequence_output = outputs[0]

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=sequence_output.dtype,
                                         device=self.config.device)
        with torch.no_grad():
            delta = sequence_output.min() - sequence_output.max() - 1.0
        attention_mask = attention_mask.unsqueeze(-1)
        vectors = sequence_output + (1.0 - attention_mask) * delta
        vectors, _ = vectors.max(axis=1)

        outputs = (vectors, ) + outputs[1:]
        return outputs


class VectorText(BERTBase, LMMixin):
    config_cls = VectorTextBERTConfig

    def __init__(self, config):
        super().__init__(config)
        if config.model_type == 'first':
            embeddings = VectorTextFirstEmbeddings
        elif config.model_type == 'last':
            embeddings = VectorTextLastEmbeddings
        elif config.model_type == 'inside':
            embeddings = VectorTextInsideEmbeddings
        else:
            raise ValueError(f'{config.model_type} not recognized. `model_type`'
                             f' should be set to either `first`, `last`, '
                             f'or `inside`')

        self.embedding = embeddings(config.vocab_size,
                                    config.hidden_size,
                                    config.max_position_embedding,
                                    config.device,
                                    config.padding_idx,
                                    config.dropout_prob,
                                    config.layer_norm_eps)
        self.encoder = BertEncoder(config.num_hidden_layers,
                                   config.num_attention_heads,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   config.is_decoder,
                                   config.dropout_prob,
                                   config.hidden_act,
                                   config.layer_norm_eps,
                                   config.cross_layer_parameter_sharing,
                                   config.output_attentions,
                                   config.output_hidden_states)
        if config.use_cut_head:
            self.lm_head = LMHeadCut(config.hidden_size,
                                     config.vocab_size,
                                     config.ignore_index)
        else:
            self.lm_head = LMHead(config.hidden_size, config.vocab_size)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def init_weights(self):
        super().init_weights()

        if self.config.tie_embedding_vectors:
            self.tie_weights()

    def forward(self,
                input_ids,
                vectors,
                attention_mask=None,
                input_pos=None,
                masked_lm_labels=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        if self.config.model_type == 'inside':
            hidden_states = self.embedding(input_ids, input_pos, vectors)
        else:
            hidden_states = self.embedding(input_ids, vectors)

        if attention_mask is None:
            if self.config.model_type == 'inside':
                attention_mask = get_attention_mask(input_ids)
            else:
                max_length = max([len(x) for x in input_ids])
                attention_mask = [[1.0] * (len(x) + 1) +
                                  [0.0] * (max_length - len(x))
                                  for x in input_ids]

        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        outputs = self.encoder(hidden_states,
                               attention_mask,
                               head_mask,
                               encoder_hidden_states,
                               encoder_attention_mask)

        sequence_output = outputs[0]
        if masked_lm_labels is not None:
            max_length = max([len(x) for x in masked_lm_labels])
            if self.config.model_type == 'first':
                masked_lm_labels = [
                    [self.config.ignore_index] + x +
                    [self.config.ignore_index] * (max_length - len(x))
                    for x in masked_lm_labels
                ]
            elif self.config.model_type == 'last':
                raise NotImplementedError
            else:
                masked_lm_labels = [
                    x + [self.config.ignore_index] * (max_length - len(x))
                    for x in masked_lm_labels
                ]
            masked_lm_labels = torch.tensor(masked_lm_labels,
                                            dtype=torch.long,
                                            device=self.config.device)

        if self.config.use_cut_head:
            prediction_scores, masked_lm_labels = self.lm_head(sequence_output,
                                                               masked_lm_labels)
        else:
            prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, ) + outputs[1:]

        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.config.ignore_index)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            outputs = (masked_lm_loss,) + outputs

        return outputs


class TextVectorMeanVAE(TextVectorMean):
    config_cls = TextVectorVAEConfig

    def __init__(self, config):
        super().__init__(config)
        if config.vae_type == 'normal_tanh_abs':
            self.vae_head = VAENormalTanhAbs(config.hidden_size)
        else:
            raise ValueError(f'{config.vae_type} not recognized. `vae_type`'
                             f' should be set to either `normal_tanh_abs`')

    def forward(self,
                input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
        vectors = outputs[0]
        statistics = self.vae_head(vectors)
        outputs = (statistics, ) + outputs[1:]
        return outputs
