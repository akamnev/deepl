import torch
import torch.nn as nn

__all__ = [
    'WordEmbeddings',
    'AbsolutePositionEmbeddings',
    'VectorTextFirstEmbeddings',
    'VectorTextLastEmbeddings',
    'VectorTextInsideEmbeddings'
]


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size,
                 device='cpu', padding_idx=0,
                 layer_norm_eps=1e-12, dropout_prob=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.device = device
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        max_length = max([len(x) for x in input_ids])
        input_ids = [x + [self.padding_idx] * (max_length - len(x))
                     for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long,
                                 device=self.device)
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AbsolutePositionEmbeddingsBase(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embedding,
                 device='cpu', padding_idx=0,
                 dropout_prob=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.padding_idx = padding_idx
        self.device = device
        max_position_embedding += padding_idx + 1
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embedding,
                                                hidden_size,
                                                padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)


class AbsolutePositionEmbeddings(AbsolutePositionEmbeddingsBase):
    def forward(self, input_ids):
        max_length = max([len(x) for x in input_ids])
        position_ids = [[xi for xi in range(self.padding_idx + 1,
                                            len(x) + self.padding_idx + 1)]
                        + [self.padding_idx] * (max_length - len(x))
                        for x in input_ids]
        input_ids = [x + [self.padding_idx] * (max_length - len(x))
                     for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long,
                                 device=self.device)
        position_ids = torch.tensor(position_ids, dtype=torch.long,
                                    device=self.device)
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VectorTextFirstEmbeddings(AbsolutePositionEmbeddingsBase):
    """Add the text's vector to the first token"""
    def forward(self, input_ids, vectors):
        max_length = max([len(x) for x in input_ids])
        position_ids = [[xi for xi in range(self.padding_idx + 1,
                                            len(x) + self.padding_idx + 2)]
                        + [self.padding_idx] * (max_length - len(x))
                        for x in input_ids]
        input_ids = [x + [self.padding_idx] * (max_length - len(x))
                     for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long,
                                 device=self.device)
        position_ids = torch.tensor(position_ids, dtype=torch.long,
                                    device=self.device)
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        vectors = torch.as_tensor(vectors, device=self.device,
                                  dtype=inputs_embeds.dtype)
        vectors = vectors.unsqueeze(dim=1)
        inputs_embeds = torch.cat((vectors, inputs_embeds), dim=1)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class VectorTextLastEmbeddings(AbsolutePositionEmbeddingsBase):
    """Add the text's vector to the last token"""
    def forward(self, input_ids, vectors):
        max_length = max([len(x) for x in input_ids]) + 1
        position_ids = []
        inputs_embeds = []
        for n, seq in enumerate(input_ids):
            len_seq = len(seq)
            position_ids.append(
                list(range(self.padding_idx + 1,
                           len_seq + self.padding_idx + 2))
                + [self.padding_idx] * (max_length - len_seq - 1))
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
        inputs_embeds = torch.cat(inputs_embeds, dim=0)
        position_ids = torch.tensor(position_ids, dtype=torch.long,
                                    device=self.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VectorTextInsideEmbeddings(AbsolutePositionEmbeddingsBase):
    def forward(self, input_ids, input_pos, vectors):
        max_length = max([len(x) for x in input_ids])
        position_ids = []
        inputs_embeds = []
        for n, (seq, pos) in enumerate(zip(input_ids, input_pos)):
            pos.sort()
            position_ids.append(
                list(range(self.padding_idx + 1,
                           len(seq) + self.padding_idx + 1))
                + [self.padding_idx] * (max_length - len(seq)))
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

        inputs_embeds = torch.cat(inputs_embeds, dim=0)
        position_ids = torch.tensor(position_ids, dtype=torch.long,
                                    device=self.device)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
