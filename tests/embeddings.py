import numpy as np
import torch
from deepl.layers.embeddings import *


def test_word_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = WordEmbeddings(vocab_size=10, hidden_size=5)
    w = emb(input_ids)
    print()
    print(w)


def test_text_first_vector_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = VectorFirstEmbeddings(vocab_size=10, hidden_size=5)
    vector = np.array([[0.1, -0.1, 0.2, -0.15, 0.25],
                       [0.1, -0.1, 0.2, -0.15, 0.25]])
    w = emb(input_ids, vector)
    print()
    print(w)

    vector = torch.tensor(vector)
    w = emb(input_ids, vector)
    print()
    print(w)


def test_text_last_vector_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = VectorLastEmbeddings(vocab_size=10, hidden_size=5)
    vector = np.array([[0.1, -0.1, 0.2, -0.15, 0.25],
                       [0.1, -0.1, 0.2, -0.15, 0.25]])
    w = emb(input_ids, vector)
    print()
    print(w)

    vector = torch.tensor(vector)
    print(vector.dtype)
    w = emb(input_ids, vector)
    print()
    print(w)


def test_mlm_vector_embeddings():
    input_ids = [[1, 2], [1, 2, 3], [1], [1, 2, 3]]
    input_pos = [[0], [0, 2], [], [1]]
    emb = VectorInsideEmbeddings(vocab_size=10, hidden_size=5)
    vector = np.array([[0.1, -0.1, 0.2, -0.15, 0.25],
                       [0.1, -0.1, 0.2, -0.15, 0.25],
                       [0.1, -0.1, 0.2, -0.15, 0.25],
                       [0.1, -0.1, 0.2, -0.15, 0.25]])
    w = emb(input_ids, input_pos, vector)
    print()
    print(w)

    vector = torch.tensor(vector)

    w = emb(input_ids, input_pos, vector)
    print()
    print(w)
