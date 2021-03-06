"""Bidirectional Encoder Representations from Transformers"""
import torch
from .base import BERTBase, LMMixin
from ..layers.embeddings import (WordEmbeddings,
                                 AbsolutePositionEmbeddings,
                                 VectorTextFirstEmbeddings,
                                 VectorTextLastEmbeddings,
                                 VectorTextInsideEmbeddings,
                                 VectorTextFirstAbsolutePositionEmbeddings,
                                 VectorTextLastAbsolutePositionEmbeddings,
                                 VectorTextInsideAbsolutePositionEmbeddings

)
from ..layers.encoders import BertEncoder, LMHead, LMHeadCut, GraphConvEncoder
from ..layers.utils import get_attention_mask
from .config import (BERTLanguageModelConfig,
                     VectorTextBERTConfig,
                     TextVectorVAEConfig,
                     VPP, VAEType)
from ..layers.vae import VAENormalTanhAbs


class BERT(BERTBase):
    def __init__(self, config):
        super().__init__(config)
        if config.max_position_embedding > 0:
            self.embedding = AbsolutePositionEmbeddings(config.vocab_size,
                                                        config.hidden_size,
                                                        config.max_position_embedding,
                                                        config.device,
                                                        config.padding_idx,
                                                        config.dropout_prob,
                                                        config.layer_norm_eps)
        else:
            self.embedding = WordEmbeddings(config.vocab_size,
                                            config.hidden_size,
                                            config.device,
                                            config.padding_idx,
                                            config.layer_norm_eps,
                                            config.dropout_prob)
        self.encoder = BertEncoder(config.num_hidden_layers,
                                   config.num_attention_heads,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   config.half_width_key,
                                   config.half_width_val,
                                   config.is_decoder,
                                   config.dropout_head,
                                   config.dropout_prob,
                                   config.hidden_act,
                                   config.layer_norm_eps,
                                   config.cross_layer_parameter_sharing,
                                   config.output_attentions,
                                   config.output_hidden_states)

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        hidden_states = self.embedding(input_ids)

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)

        outputs = self.encoder(hidden_states,
                               attention_mask,
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
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=self.config.ignore_index)

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
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
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
        outputs = [prediction_scores] + outputs[1:]

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            outputs = [masked_lm_loss] + outputs

        return outputs


class TextVectorMean(BERT):
    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                eps=1e-8):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
        sequence_output = outputs[0]

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=sequence_output.dtype,
                                         device=self.config.device)

        norm = attention_mask.sum(dim=1, keepdim=True) + eps
        attention_mask = attention_mask.unsqueeze(-1)
        vectors = (sequence_output * attention_mask).sum(dim=1) / norm
        outputs = [vectors] + outputs[1:]
        return outputs


class TextVectorMax(BERT):
    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
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
        vectors, _ = vectors.max(dim=1)

        outputs = [vectors] + outputs[1:]
        return outputs


class VectorText(BERTBase, LMMixin):
    config_cls = VectorTextBERTConfig

    def __init__(self, config):
        super().__init__(config)
        if config.model_type == VPP.FIRST:
            if config.max_position_embedding > 0:
                embeddings = VectorTextFirstAbsolutePositionEmbeddings
            else:
                embeddings = VectorTextFirstEmbeddings
        elif config.model_type == VPP.LAST:
            if config.max_position_embedding > 0:
                embeddings = VectorTextLastAbsolutePositionEmbeddings
            else:
                embeddings = VectorTextLastEmbeddings
        elif config.model_type == VPP.INSIDE:
            if config.max_position_embedding > 0:
                embeddings = VectorTextInsideAbsolutePositionEmbeddings
            else:
                embeddings = VectorTextInsideEmbeddings
        else:
            raise ValueError(f'{config.model_type} not recognized. `model_type`'
                             f' should be set to either `FIRST`, `LAST`, '
                             f'or `INSIDE`')
        if config.max_position_embedding > 0:
            self.embedding = embeddings(config.vocab_size,
                                        config.hidden_size,
                                        config.max_position_embedding,
                                        config.device,
                                        config.padding_idx,
                                        config.dropout_prob,
                                        config.layer_norm_eps)
        else:
            self.embedding = embeddings(config.vocab_size,
                                        config.hidden_size,
                                        config.device,
                                        config.padding_idx,
                                        config.layer_norm_eps,
                                        config.dropout_prob)
        self.encoder = BertEncoder(config.num_hidden_layers,
                                   config.num_attention_heads,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   config.half_width_key,
                                   config.half_width_val,
                                   config.is_decoder,
                                   config.dropout_head,
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
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        if self.config.model_type == VPP.INSIDE:
            hidden_states = self.embedding(input_ids, input_pos, vectors)
        else:
            hidden_states = self.embedding(input_ids, vectors)

        if attention_mask is None:
            if self.config.model_type == VPP.INSIDE:
                attention_mask = get_attention_mask(input_ids)
            else:
                max_length = max([len(x) for x in input_ids])
                attention_mask = [[1.0] * (len(x) + 1) +
                                  [0.0] * (max_length - len(x))
                                  for x in input_ids]

        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)

        outputs = self.encoder(hidden_states,
                               attention_mask,
                               encoder_hidden_states,
                               encoder_attention_mask)

        sequence_output = outputs[0]
        if masked_lm_labels is not None:
            max_length = max([len(x) for x in masked_lm_labels])
            if self.config.model_type == VPP.FIRST:
                masked_lm_labels = [
                    [self.config.ignore_index] + x +
                    [self.config.ignore_index] * (max_length - len(x))
                    for x in masked_lm_labels
                ]
            elif self.config.model_type == VPP.LAST:
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
        outputs = [prediction_scores] + outputs[1:]

        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=self.config.ignore_index)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            outputs = [masked_lm_loss] + outputs

        return outputs


class TextVectorMeanVAE(TextVectorMean):
    config_cls = TextVectorVAEConfig

    def __init__(self, config):
        super().__init__(config)
        if config.vae_type == VAEType.NORMAL_TANH_ABS:
            self.vae_head = VAENormalTanhAbs(config.hidden_size)
        else:
            raise ValueError(f'{config.vae_type} not recognized. `vae_type`'
                             f' should be set to either `normal_tanh_abs`')

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                eps=1e-8):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  eps=eps)
        vectors = outputs[0]
        statistics = self.vae_head(vectors)
        outputs = [statistics] + outputs[1:]
        return outputs


class GraphConvBase(BERTBase):
    def __init__(self, config):
        super().__init__(config)
        if config.max_position_embedding > 0:
            self.embedding = AbsolutePositionEmbeddings(config.vocab_size,
                                                        config.hidden_size,
                                                        config.max_position_embedding,
                                                        config.device,
                                                        config.padding_idx,
                                                        config.dropout_prob,
                                                        config.layer_norm_eps)
        else:
            self.embedding = WordEmbeddings(config.vocab_size,
                                            config.hidden_size,
                                            config.device,
                                            config.padding_idx,
                                            config.layer_norm_eps,
                                            config.dropout_prob)
        self.encoder = GraphConvEncoder(
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            half_width_key=config.half_width_key,
            half_width_val=config.half_width_val,
            dropout_head=config.dropout_head,
            dropout_prob=config.dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            cross_layer_parameter_sharing=config.cross_layer_parameter_sharing,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states)

    def forward(self,
                input_ids,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        hidden_states = self.embedding(input_ids)

        if attention_mask is None:
            attention_mask = get_attention_mask(input_ids)
        attention_mask = torch.as_tensor(attention_mask,
                                         dtype=hidden_states.dtype,
                                         device=self.config.device)

        outputs = self.encoder(hidden_states,
                               attention_mask,
                               encoder_hidden_states,
                               encoder_attention_mask)
        return outputs


class GraphConvLanguageModel(GraphConvBase, LMMixin):
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
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=self.config.ignore_index)

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
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
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
        outputs = [prediction_scores] + outputs[1:]

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            outputs = [masked_lm_loss] + outputs

        return outputs
