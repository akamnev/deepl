import torch
import torch.nn as nn
from .dropout import VariationalNormalEpanechnikovDropout

__all__ = [
    'WordEmbeddings',
    'VectorFirstEmbeddings',
    'VectorLastEmbeddings',
    'VectorInsideEmbeddings',
]


class WordEmbeddingsBase(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position=0,
                 padding_idx=0,
                 device='cpu'):
        super().__init__()
        self.padding_idx = padding_idx
        self.device = device if isinstance(device, torch.device) \
            else torch.device(device)
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)
        self.max_position = max_position
        if max_position > 0:
            self.position_embeddings = nn.Embedding(max_position + 1,
                                                    hidden_size,
                                                    padding_idx=padding_idx)

        self.dropout = VariationalNormalEpanechnikovDropout(hidden_size)


class WordEmbeddings(WordEmbeddingsBase):
    def forward(self, input_ids):
        max_length = max([len(x) for x in input_ids])
        position_embeddings = None
        if self.max_position > 0:
            position_ids = [[xi for xi in range(self.padding_idx + 1, len(x) + self.padding_idx + 1)]
                            + [self.padding_idx] * (max_length - len(x))
                            for x in input_ids]
            position_ids = torch.tensor(position_ids, dtype=torch.long,
                                        device=self.device)
            position_embeddings = self.position_embeddings(position_ids)

        input_ids = [x + [self.padding_idx] * (max_length - len(x))
                     for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long,
                                 device=self.device)
        embeddings = self.word_embeddings(input_ids)
        if position_embeddings is not None:
            embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class VectorEmbeddingsBase(WordEmbeddingsBase):
    def forward(self, input_ids, vectors, input_pos):
        raise NotImplementedError


class VectorFirstEmbeddings(VectorEmbeddingsBase):
    def forward(self, input_ids, vectors, input_pos=None):
        max_length = max([len(x) for x in input_ids])

        position_embeddings = None
        if self.max_position > 0:
            position_ids = [[xi for xi in range(self.padding_idx + 1, len(x) + self.padding_idx + 1)]
                            + [self.padding_idx] * (max_length - len(x))
                            for x in input_ids]
            position_ids = torch.tensor(position_ids, dtype=torch.long,
                                        device=self.device)
            position_embeddings = self.position_embeddings(position_ids)

        input_ids = [x + [self.padding_idx] * (max_length - len(x))
                     for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long,
                                 device=self.device)
        inputs_embeds = self.word_embeddings(input_ids)

        if position_embeddings is not None:
            inputs_embeds = inputs_embeds + position_embeddings

        vectors = torch.as_tensor(vectors, device=self.device,
                                  dtype=inputs_embeds.dtype)
        vectors = vectors.unsqueeze(dim=1)
        embeddings = torch.cat((vectors, inputs_embeds), dim=1)
        embeddings = self.dropout(embeddings)
        return embeddings


class VectorLastEmbeddings(VectorEmbeddingsBase):
    def forward(self, input_ids, vectors, input_pos):
        # TODO: add position embeddings
        raise NotImplementedError

        max_length = max([len(x) for x in input_ids]) + 1
        inputs_embeds = []
        for n, seq in enumerate(input_ids):
            len_seq = len(seq)
            seq = torch.tensor(seq, dtype=torch.long, device=self.device)
            seq_embeds = self.word_embeddings(seq)
            seq_pad = torch.zeros((max_length - len_seq - 1, seq_embeds.shape[1]),
                                  device=self.device, dtype=seq_embeds.dtype)
            v = torch.as_tensor(vectors[n], device=self.device,
                                dtype=seq_embeds.dtype)
            seq_embeds_with_vector = torch.cat((seq_embeds,
                                                v.view(1, -1),
                                                seq_pad), dim=0)
            seq_embeds_with_vector = seq_embeds_with_vector.unsqueeze(0)
            inputs_embeds.append(seq_embeds_with_vector)
        embeddings = torch.cat(inputs_embeds, dim=0)
        embeddings = self.dropout(embeddings)
        return embeddings


class VectorInsideEmbeddings(VectorEmbeddingsBase):
    def forward(self, input_ids, vectors, input_pos):
        max_length = max([len(x) for x in input_ids])

        position_embeddings = None
        if self.max_position > 0:
            position_ids = [[xi for xi in range(self.padding_idx + 1, len(x) + self.padding_idx + 1)]
                            + [self.padding_idx] * (max_length - len(x))
                            for x in input_ids]
            position_ids = torch.tensor(position_ids, dtype=torch.long,
                                        device=self.device)
            position_embeddings = self.position_embeddings(position_ids)

        inputs_embeds = []
        for n, (seq, pos) in enumerate(zip(input_ids, input_pos)):
            pos.sort()
            seq.extend([self.padding_idx] * (max_length - len(seq)))

            seq_embeds = []
            pmo = 0
            for p in pos:
                s = torch.tensor(seq[pmo:p], dtype=torch.long,
                                 device=self.device)
                e = self.word_embeddings(s)
                v = torch.as_tensor(vectors[n], device=self.device,
                                    dtype=e.dtype)
                seq_embeds.append(e)
                seq_embeds.append(v.view(1, -1))
                pmo = p + 1
            if pmo < len(seq):
                s = torch.tensor(seq[pmo:], dtype=torch.long,
                                 device=self.device)
                e = self.word_embeddings(s)
                seq_embeds.append(e)

            seq_embeds = torch.cat(seq_embeds, dim=0)
            seq_embeds = seq_embeds.unsqueeze(0)
            inputs_embeds.append(seq_embeds)

        embeddings = torch.cat(inputs_embeds, dim=0)

        if position_embeddings is not None:
            embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings
