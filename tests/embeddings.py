import pytest
import numpy as np
import torch
from layers.embeddings import *


def test_word_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = WordEmbeddings(vocab_size=10, hidden_size=5)
    w = emb(input_ids)
    print()
    print(w)


def test_abs_pos_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = AbsolutePositionEmbeddings(vocab_size=10, hidden_size=5,
                                     max_position_embedding=10)
    w = emb(input_ids)
    print()
    print(w)


def test_text_first_vector_embeddings():
    input_ids = [[1, 2], [1, 2, 3]]
    emb = VectorTextFirstEmbeddings(vocab_size=10, hidden_size=5,
                                    max_position_embedding=10)
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
    emb = VectorTextLastEmbeddings(vocab_size=10, hidden_size=5,
                                   max_position_embedding=10)
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
    emb = VectorTextInsideEmbeddings(vocab_size=10, hidden_size=5,
                                     max_position_embedding=10)
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
