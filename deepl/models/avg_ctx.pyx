from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef extern from "../src/avg_ctx.h" namespace "AvgCtxModel":

    cdef vector[pair[int, int]] single_group_spans(vector[int] pos)
    cdef pair[vector[vector[int]], vector[vector[int]]] single_context(
            vector[pair[int, int]] spans, int hw, int seq_length
    )
    void single_mix_token_sparse_matrix(
            vector[pair[int, int]]& spans,
            pair[vector[vector[int]], vector[vector[int]]]& ctx,
            int max_seq_length,
            vector[int] &row, vector[int] &col, vector[float] &value
    )

    void single_sequence_impl(
            vector[int] pos, int seq_length, int hw, int max_seq_length,
            vector[int] &row, vector[int] &col, vector[float] &value
    )

    void batch_sequence_impl(
            vector[vector[int]] pos, vector[int] seq_length,
            int hw, int max_seq_length, int n_threads,
            vector[vector[int]] &row, vector[vector[int]] &col, vector[vector[float]] &value
    )

def _group_spans(x):
    return single_group_spans(x)

def _single_context(spans, hw, seq_length):
    return single_context(spans, hw, seq_length)

def _single_mix_token_sparse_matrix(spans, ctx, max_seq_length):
    cdef vector[int] row
    cdef vector[int] col
    cdef vector[float] value
    single_mix_token_sparse_matrix(spans, ctx, max_seq_length, row, col, value)
    return row, col, value

def single_sequence(pos, seq_length, hw, max_seq_length):
    cdef vector[int] row
    cdef vector[int] col
    cdef vector[float] value
    single_sequence_impl(pos, seq_length, hw, max_seq_length, row, col, value)
    return row, col, value

def batch_sequence(pos, seq_length, hw, max_seq_length, n_threads=-1):
    cdef vector[vector[int]] row
    cdef vector[vector[int]] col
    cdef vector[vector[float]] value
    assert len(pos) == len(seq_length)
    batch_sequence_impl(pos, seq_length, hw, max_seq_length, n_threads, row, col, value)
    return row, col, value
