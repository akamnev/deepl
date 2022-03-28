import pytest
import torch
from deepl.train.sample import mask_token_with_true_rand_mlm
from deepl.models.avg_ctx import _group_spans, _single_context, \
    single_sequence, batch_sequence


@pytest.fixture
def single_example():
    pos = [0, 5, 9, 6]
    hw = 3
    seq_len = 10
    max_seq_len = 17
    return pos, seq_len, max_seq_len, hw


@pytest.fixture
def batch_example():
    pos = [[0, 1, 2], [], [4], [3, 2], [1, 4, 2], [], [4], [3, 2], [1, 2, 4]]
    seq_len = [5, 6, 7, 7, 7, 7, 7, 7, 7]
    hw = 3
    max_seq_len = 7

    return pos, seq_len, max_seq_len, hw


class AvgCtxTokenModelPy:
    def __init__(self, hw):
        self.hw = hw

    @staticmethod
    def group_spans(pos):
        if len(pos) == 0:
            spans = []
        elif len(pos) == 1:
            spans = [(pos[0], pos[0] + 1)]
        else:
            pos = sorted(pos)
            spans = []
            i1, i2 = pos[0], pos[0]
            for i in range(1, len(pos)):
                if i2 + 1 == pos[i]:
                    i2 = pos[i]
                else:
                    spans.append((i1, i2 + 1))
                    i1 = i2 = pos[i]
            spans.append((i1, i2 + 1))
        return spans

    def context(self, spans, seq_len):
        ctx_left, ctx_right = [], []
        for i, (i1, i2) in enumerate(spans):
            cl = list(range(max(i1 - self.hw, 0), i1))
            cr = list(range(i2, min(seq_len, i2 + self.hw)))
            # move to left
            for j in range(i - 1, -1, -1):
                ij_1, ij_2 = spans[j]
                if ij_2 <= cl[0]:
                    break
                else:
                    cl = [idx for idx in cl if not (ij_1 <= idx < ij_2)]
            # move to right
            for j in range(i + 1, len(spans)):
                ij_1, ij_2 = spans[j]
                if cr[-1] < ij_1:
                    break
                else:
                    cr = [idx for idx in cr if not (ij_1 <= idx < ij_2)]

            ctx_left.append(cl)
            ctx_right.append(cr)
        return ctx_left, ctx_right

    @staticmethod
    def mix_token_matrix(spans, ctx, max_ids_len):
        ctx_left, ctx_right = ctx
        avg_token_model = torch.eye(max_ids_len, dtype=torch.float32)
        for n in range(len(spans)):
            i1, i2 = spans[n]
            cl, cr = ctx_left[n], ctx_right[n]

            span_width = i2 - i1
            avg_token_model[i1:i2] = 0.0  # remove masked tokens
            cl_width = len(cl)
            cr_width = len(cr)
            if cl and cr:
                for i in range(i1, i2):
                    for j in cl:
                        avg_token_model[i, j] = (i - i1 + 1) / (cl_width * (span_width + 1))
                    for j in cr:
                        avg_token_model[i, j] = (i2 - i) / (cr_width * (span_width + 1))
            elif cl and not cr:
                for i in range(i1, i2):
                    for j in cl:
                        avg_token_model[i, j] = 1.0 / cl_width
            else:
                for i in range(i1, i2):
                    for j in cr:
                        avg_token_model[i, j] = 1.0 / cr_width
        return avg_token_model

    def __call__(self, pos, seq_len, max_seq_len):
        """

        Args:
            pos: list(list(int)) - batch of position where mask token is inserted
            seq_len: list(list(int))
            max_seq_len: int
        Returns:
            (torch.Tensor)
        """
        spans = [self.group_spans(p) for p in pos]
        ctx = [self.context(spans[i], seq_len[i]) for i in range(len(spans))]
        mix_matrix = [
            self.mix_token_matrix(spans[i], ctx[i], max_seq_len)
            for i in range(len(spans))
        ]
        return mix_matrix


def create_sparse_tensor(row, col, value, max_seq_len):
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    value = torch.tensor(value, dtype=torch.float32)

    idx = torch.cat([row.view(1, -1), col.view(1, -1)])
    mix = torch.sparse_coo_tensor(idx, value, (max_seq_len, max_seq_len))
    return mix


def test_avg_mask_token():
    text_ids = [list(range(11, 21))]

    text_ids, labels, pos = mask_token_with_true_rand_mlm(
        tokens=text_ids,
        proba_mask=0.15,
        proba_true_token=0.1,
        proba_random_token=0.1,
        random_token_range=(10, 1000),
        id_mask=1,
        id_ignore=-100
    )

    max_text_len = len(text_ids[0])
    obj = AvgCtxTokenModelPy(hw=5)

    # group by spans
    text_ids_0 = text_ids[0]
    pos_0 = sorted(pos[0])
    spans = []
    if pos_0:
        i1 = i2 = pos_0[0]
        for i in range(1, len(pos_0)):
            if i2 + 1 == pos_0[i]:
                i2 = pos_0[i]
            else:
                spans.append((i1, i2 + 1))
                i1 = i2 = pos_0[i]
        spans.append((i1, i2 + 1))
    else:
        spans.append([])
    # CHECK SPANS
    obj_spans = obj.group_spans(pos[0])
    assert spans == obj_spans
    # для каждого интервала определяем правый и левые контексты
    hw = obj.hw
    ctx_left, ctx_right = [], []
    for i, (i1, i2) in enumerate(spans):
        cl = list(range(max(i1 - hw, 0), i1))
        cr = list(range(i2, min(len(text_ids_0), i2 + hw)))
        # удаляем токены перекрытия
        # move to left
        for j in range(i - 1, -1, -1):
            ij_1, ij_2 = spans[j]
            if ij_2 <= cl[0]:
                break
            else:
                cl = [idx for idx in cl if not (ij_1 <= idx < ij_2)]
        # move to right
        for j in range(i + 1, len(spans)):
            ij_1, ij_2 = spans[j]
            if cr[-1] < ij_1:
                break
            else:
                cr = [idx for idx in cr if not (ij_1 <= idx < ij_2)]

        ctx_left.append(cl)
        ctx_right.append(cr)
    # CHECK CONTEXT
    obj_ctx = obj.context(spans, text_ids_0)
    assert ctx_left == obj_ctx[0]
    assert ctx_right == obj_ctx[1]
    # строим torch матрицу внимания
    avg_token_model_0 = torch.eye(max_text_len, dtype=torch.float32)
    for n in range(len(spans)):
        i1, i2 = spans[n]
        cl, cr = ctx_left[n], ctx_right[n]

        span_width = i2 - i1
        avg_token_model_0[i1:i2] = 0.0  # remove masked tokens
        cl_width = len(cl)
        cr_width = len(cr)
        if cl and cr:
            for i in range(i1, i2):
                for j in cl:
                    avg_token_model_0[i, j] = (i - i1 + 1) / (cl_width * (span_width + 1))
                for j in cr:
                    avg_token_model_0[i, j] = (i2 - i) / (cr_width * (span_width + 1))
        elif cl and not cr:
            for i in range(i1, i2):
                for j in cl:
                    avg_token_model_0[i, j] = 1.0 / cl_width
        else:
            for i in range(i1, i2):
                for j in cr:
                    avg_token_model_0[i, j] = 1.0 / cr_width
    # CHECK MIX MATRIX
    obj_avg_token_model = obj.mix_token_matrix(spans, (ctx_left, ctx_right), max_text_len)
    ans = torch.all(torch.isclose(avg_token_model_0, obj_avg_token_model))
    assert True


def test_run_avg_mask_token_batch_py():
    text_len = 20
    text_ids = [
        list(range(11, 11 + text_len)),
        list(range(21, 21 + text_len))
    ]

    text_ids, labels, pos = mask_token_with_true_rand_mlm(
        tokens=text_ids,
        proba_mask=0.15,
        proba_true_token=0.1,
        proba_random_token=0.1,
        random_token_range=(10, 1000),
        id_mask=1,
        id_ignore=-100
    )

    obj = AvgCtxTokenModelPy(hw=5)
    seq_len = list(map(len, text_ids))
    max_seq_len = max(seq_len)
    ans = obj(pos, seq_len, max_seq_len)
    print(ans)


def test_group_span_cpp_impl(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    obj = AvgCtxTokenModelPy(hw=hw)

    spans = _group_spans(pos)
    spans_py = obj.group_spans(pos)

    assert spans == spans_py


def test_group_span_cpp_impl_empty(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    pos = []
    obj = AvgCtxTokenModelPy(hw=hw)

    spans = _group_spans(pos)
    spans_py = obj.group_spans(pos)

    assert spans == spans_py


def test_ctx_cpp_impl(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    obj = AvgCtxTokenModelPy(hw=hw)

    spans = _group_spans(pos)
    ctx_left, ctx_right = _single_context(spans, hw, seq_len)

    spans_py = obj.group_spans(pos)
    ctx_left_py, ctx_right_py = obj.context(spans_py, seq_len)

    assert ctx_left == ctx_left_py
    assert ctx_right == ctx_right_py


def test_ctx_cpp_impl_empty(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    pos = []
    obj = AvgCtxTokenModelPy(hw=hw)

    spans = _group_spans(pos)
    ctx_left, ctx_right = _single_context(spans, hw, seq_len)

    spans_py = obj.group_spans(pos)
    ctx_left_py, ctx_right_py = obj.context(spans_py, seq_len)

    assert ctx_left == ctx_left_py
    assert ctx_right == ctx_right_py


def test_cpp_single_impl(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    obj = AvgCtxTokenModelPy(hw=hw)

    row, col, value = single_sequence(pos, seq_len, hw, max_seq_len)

    mix = create_sparse_tensor(row, col, value, max_seq_len)
    mix = mix.to_dense()

    mix_py = obj([pos], [seq_len], max_seq_len)
    assert len(mix_py) == 1
    mix_py = mix_py[0]

    assert torch.all(torch.isclose(mix, mix_py))


def test_cpp_single_empty_impl(single_example):
    pos, seq_len, max_seq_len, hw = single_example
    pos = []
    obj = AvgCtxTokenModelPy(hw=hw)

    row, col, value = single_sequence(pos, seq_len, hw, max_seq_len)

    mix = create_sparse_tensor(row, col, value, max_seq_len)
    mix = mix.to_dense()

    mix_py = obj([pos], [seq_len], max_seq_len)
    assert len(mix_py) == 1
    mix_py = mix_py[0]

    assert torch.all(torch.isclose(mix, mix_py))


def test_cpp_batch_impl(batch_example):
    pos, seq_len, max_seq_len, hw = batch_example
    row, col, value = batch_sequence(pos, seq_len, hw, max_seq_len, n_threads=-1)
    obj = AvgCtxTokenModelPy(hw=hw)

    mix = [
        create_sparse_tensor(row[i], col[i], value[i], max_seq_len)
        for i in range(len(pos))
    ]
    mix = [m.to_dense() for m in mix]

    mix_py = obj(pos, seq_len, max_seq_len)

    assert len(mix) == len(mix_py)

    for i in range(len(mix)):
        assert torch.all(torch.isclose(mix[i], mix_py[i]))
