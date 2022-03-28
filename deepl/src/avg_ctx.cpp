#include <thread>
#include <iostream>
#include "avg_ctx.h"

namespace AvgCtxModel {
    /* Входные данные вектор целых чисел где токены
     заменены на маску. Пример: [1, 2, 4, 5 9]
    */
    std::vector<std::pair<int, int>> single_group_spans(std::vector<int>& pos) {
        std::vector<std::pair<int, int>> spans;
        if (pos.size() == 0)
            return spans;
        else if (pos.size() == 1) {
            spans.emplace_back(std::pair<int, int>{pos[0], pos[0] + 1});
            return spans;
        }
        std::sort(pos.begin(), pos.end());

        int i1 = pos[0];
        int i2 = i1;

        for (int i = 1; i < pos.size(); i++) {
            if (i2 + 1 == pos[i])
                i2 = pos[i];
            else {
                spans.emplace_back(std::pair < int, int > {i1, i2 + 1});
                i1 = pos[i];
                i2 = i1;
            }
        }
        spans.emplace_back(std::pair < int, int > {i1, i2 + 1});
        return spans;
    };

    /* Функция вычисляет контекст
     * Входные данные [(0, 2), (5, 6)] & 10
     * Выходные данные ([[], [2, 3, 4]], [[2, 3, 4], [6, 7, 8])
     * */
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> single_context(
            std::vector<std::pair<int, int>>& spans, int hw, int seq_length) {

        std::vector<std::vector<int>> ctx_left, ctx_right;

        for (int i = 0; i < spans.size(); i++) {
            int i1 = spans[i].first;
            int i2 = spans[i].second;

            std::vector<int> cl, cr;  // left & right contexts
            // init left context
            for (int j = std::max(i1 - hw, 0); j < i1; j++)
                cl.push_back(j);
            // init right context
            for (int j = i2; j < std::min(seq_length, i2 + hw); j++)
                cr.push_back(j);

            // move to left
            for (int j = i - 1; j >= 0; j--) {
                int ij_1 = spans[j].first;
                int ij_2 = spans[j].second;
                if (ij_2 <= cl[0])
                    break;
                else {
                    std::vector<int> cl_new;
                    for (auto it: cl) {
                        if (ij_1 > it || it >= ij_2) {
                            cl_new.push_back(it);
                        }
                    }
                    cl = cl_new;
                }
            }
            // move to right
            for (int j = i + 1; j < spans.size(); j++) {
                int ij_1 = spans[j].first;
                int ij_2 = spans[j].second;
                if (cr.back() < ij_1)
                    break;
                else {
                    std::vector<int> cr_new;
                    for (auto it : cr) {
                        if (ij_1 > it || it >= ij_2) {
                            cr_new.push_back(it);
                        }
                    }
                    cr = cr_new;
                }
            }
            ctx_left.emplace_back(cl);
            ctx_right.emplace_back(cr);
        }
        return std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> {ctx_left, ctx_right};
    }

    void single_mix_token_sparse_matrix(
            std::vector<std::pair<int, int>>& spans,
            std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>& ctx,
            int max_seq_length,
            std::vector<int> &row, std::vector<int> &col, std::vector<float> &value) {

        auto ctx_left = ctx.first;
        auto ctx_right = ctx.second;

        // init diagonal
        int idx_last = 0;
        for (int n = 0; n < spans.size(); n++) {
            int i1 = spans[n].first;
            int i2 = spans[n].second;
            for (int i = idx_last; i < i1; i++) {
                row.push_back(i);
                col.push_back(i);
                value.push_back(1.0);
            }
            idx_last = i2;
        }
        for (int i = idx_last; i < max_seq_length; i++) {
            row.push_back(i);
            col.push_back(i);
            value.push_back(1.0);
        }
        // end init diagonal
        // fill non-diagonal
        for (int n = 0; n < spans.size(); n++) {
            int i1 = spans[n].first;
            int i2 = spans[n].second;
            auto cl = ctx_left[n];
            auto cr = ctx_right[n];

            int span_width = i2 - i1;
            int cl_width = cl.size();
            int cr_width = cr.size();
            float denom;
            if (cl_width > 0 && cr_width > 0) {
                for (int i = i1; i < i2; i++) {
                    denom = (float) cl_width * (span_width + 1);
                    for (auto j : cl) {
                        row.push_back(i);
                        col.push_back(j);
                        value.push_back(float(i - i1 + 1) / denom);
                    }
                    denom = (float) cr_width * (span_width + 1);
                    for (auto j : cr) {
                        row.push_back(i);
                        col.push_back(j);
                        value.push_back(float(i2 - i) / denom);
                    }
                }
            }
            else if (cl_width > 0) {
                for (int i = i1; i < i2; i++) {
                    denom = (float) cl_width;
                    for (auto j : cl) {
                        row.push_back(i);
                        col.push_back(j);
                        value.push_back(1.0 / denom);
                    }
                }
            }
            else {
                for (int i = i1; i < i2; i++) {
                    denom = (float ) cr_width;
                    for (auto j : cr) {
                        row.push_back(i);
                        col.push_back(j);
                        value.push_back(1.0 / denom);
                    }
                }
            }
        }
    }
    void single_sequence_impl(
            std::vector<int> pos, int seq_length, int hw, int max_seq_length,
            std::vector<int> &row, std::vector<int> &col, std::vector<float> &value) {
        auto spans = single_group_spans(pos);
        auto ctx = single_context(spans, hw, seq_length);
        single_mix_token_sparse_matrix(spans, ctx, max_seq_length, row, col, value);
    }

    void batch_sequence_impl(
            std::vector<std::vector<int>> pos, std::vector<int> seq_length,
            int hw, int max_seq_length, int n_threads,
            std::vector<std::vector<int>> &row, std::vector<std::vector<int>> &col,
            std::vector<std::vector<float>> &value) {
        if (n_threads == -1) {
            n_threads = static_cast<int>(std::thread::hardware_concurrency());
        }
        n_threads = std::min(8, std::max(1, n_threads));

        if (n_threads == 1 || pos.size() < 2 * n_threads) {
            for (int s = 0; s < pos.size(); s++) {
                std::vector<int> row_seq;
                std::vector<int> col_seq;
                std::vector<float> value_seq;
                single_sequence_impl(pos[s], seq_length[s], hw, max_seq_length, row_seq, col_seq, value_seq);
                row.emplace_back(row_seq);
                col.emplace_back(col_seq);
                value.emplace_back(value_seq);
            }
        } else {
            std::vector<std::thread> threads;
            row.resize(pos.size());
            col.resize(pos.size());
            value.resize(pos.size());

            for (int i = 0; i < n_threads; i++) {
                threads.emplace_back(
                        [&](int this_thread) {
                            int tasks_for_thread = (pos.size() - 1) / n_threads + 1;
                            int first_task = tasks_for_thread * this_thread;
                            int last_task = std::min(tasks_for_thread * (this_thread + 1), static_cast<int>(pos.size()));
                            for (int s = first_task; s < last_task; s++) {
                                std::vector<int> row_seq;
                                std::vector<int> col_seq;
                                std::vector<float> value_seq;
                                single_sequence_impl(pos[s], seq_length[s], hw, max_seq_length, row_seq, col_seq, value_seq);
                                row[s] = row_seq;
                                col[s] = col_seq;
                                value[s] = value_seq;
                            }
                        },
                        i);
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }
    }

};