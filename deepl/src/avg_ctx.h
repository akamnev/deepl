#pragma once

#include <vector>
#include <algorithm>

namespace AvgCtxModel {

    std::vector<std::pair<int, int>> single_group_spans(std::vector<int>& pos);
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> single_context(
            std::vector<std::pair<int, int>>& spans, int hw, int seq_length);

    void single_mix_token_sparse_matrix(
            std::vector<std::pair<int, int>>& spans,
            std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>& ctx,
            int max_seq_length,
            std::vector<int> &row, std::vector<int> &col, std::vector<float> &value);

    void single_sequence_impl(
            std::vector<int> pos, int seq_length, int hw, int max_seq_length,
            std::vector<int> &row, std::vector<int> &col, std::vector<float> &value);

    void batch_sequence_impl(
            std::vector<std::vector<int>> pos, std::vector<int> seq_length,
            int hw, int max_seq_length, int n_threads,
            std::vector<std::vector<int>> &row, std::vector<std::vector<int>> &col,
            std::vector<std::vector<float>> &value);

}  // AvgCtxModel