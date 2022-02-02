"""Tests for multi_head_attention
add keyword -s to disable all capturing
"""
import pytest
import random
import time
import torch
import torch.nn.functional as F
from deepl.layers.multi_head_attention import \
    fnMultiHeadGlobalAttention, MultiHeadLocalAttention


def multi_head_global_attention_function_py(query, key, value, attention_mask):
    score = torch.matmul(query, key.transpose(-1, 2))
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    if not torch.all(attention_mask):
        lv = torch.finfo(score.dtype).min
        score += lv * (~attention_mask)
    proba = F.softmax(score, dim=-1)
    context = torch.matmul(proba, value)
    return context, proba, score


def multi_head_local_attention_function_py(query, key, value, attention_mask, hw):
    score = torch.matmul(query, key.transpose(-1, 2))
    extended_attention_mask = torch.ones_like(score, dtype=torch.bool)

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    extended_attention_mask *= attention_mask.unsqueeze(1).unsqueeze(1)
    extended_attention_mask *= attention_mask.unsqueeze(-1).unsqueeze(1)

    ones = torch.ones_like(score, dtype=torch.bool)
    ma = torch.triu(ones, diagonal=-hw)
    mb = torch.triu(ones, diagonal=hw + 1)
    local_attention_mask = ma * ~ mb
    extended_attention_mask *= local_attention_mask

    lv = torch.finfo(score.dtype).min
    score = score + lv * (~extended_attention_mask)

    proba = F.softmax(score, dim=-1)
    context = torch.matmul(proba, value)
    return context, proba


def _build_relative_index(nin, nout, device):
    q_ids = torch.arange(nout, dtype=torch.long, device=device)
    k_ids = torch.arange(nin, dtype=torch.long, device=device)
    rel_pos_ids = q_ids.view(-1, 1) - k_ids.view(1, -1)
    return rel_pos_ids


def multi_head_local_attention_position_function_py(query, key, value, pos_query, pos_key, attention_mask, hw):
    score = torch.matmul(query, key.transpose(-1, 2))
    extended_attention_mask = torch.ones_like(score, dtype=torch.bool)

    # context -> position
    score_c2p = torch.matmul(query, pos_key.transpose(-1, -2))
    index_c2p = _build_relative_index(key.shape[-2], query.shape[-2], key.device)
    index_c2p = torch.clamp(-index_c2p + hw, 0, 2 * hw)
    index_c2p = index_c2p.unsqueeze(0).unsqueeze(0)
    index_c2p = index_c2p.expand([query.shape[0], query.shape[1], query.shape[2], index_c2p.shape[-1]])
    score_c2p = torch.gather(score_c2p, dim=-1, index=index_c2p)
    # position -> context
    score_p2c = torch.matmul(key, pos_query.transpose(-1, -2))
    index_p2c = _build_relative_index(query.shape[-2], key.shape[-2], key.device)  # should change order key <-> query
    index_p2c = torch.clamp(-index_p2c + hw, 0, 2 * hw)
    index_p2c = index_p2c.unsqueeze(0).unsqueeze(0)
    index_p2c = index_p2c.expand([key.shape[0], key.shape[1], key.shape[2], index_p2c.shape[-1]])
    score_p2c = torch.gather(score_p2c, dim=-1, index=index_p2c).transpose(-1, -2)

    score = score + score_c2p + score_p2c

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    extended_attention_mask *= attention_mask.unsqueeze(1).unsqueeze(1)
    extended_attention_mask *= attention_mask.unsqueeze(-1).unsqueeze(1)

    ones = torch.ones_like(score, dtype=torch.bool)
    ma = torch.triu(ones, diagonal=-hw)
    mb = torch.triu(ones, diagonal=hw + 1)
    local_attention_mask = ma * ~ mb
    extended_attention_mask *= local_attention_mask

    lv = torch.finfo(score.dtype).min
    score = score + lv * (~extended_attention_mask)

    proba = F.softmax(score, dim=-1)
    # proba = proba * extended_attention_mask
    context = torch.matmul(proba, value)
    return context, proba


def create_example(
        batch_size_range=(64, 64),  # (64, 64) - (3, 3)
        in_token_number_range=(256, 256),  # (512, 512) - (10, 10)
        out_token_number_range=(256, 256),  # (512, 512) - (10, 10)
        head_number_range=(8, 8),  # (8, 8) (2, 2)
        head_size_range=(64, 64),  # (64, 64) (4, 4)
        hw_range=(3, 3),
        device='cpu'
):
    batch_size = random.randint(*batch_size_range)
    in_token_number = random.randint(*in_token_number_range)
    out_token_number = random.randint(*out_token_number_range)
    head_number = random.randint(*head_number_range)
    head_size = random.randint(*head_size_range)
    hw = random.randint(*hw_range)

    q = torch.rand(
        (batch_size, head_number, out_token_number, head_size),
        requires_grad=True,
        device=device
    )
    k = torch.rand(
        (batch_size, head_number, in_token_number, head_size),
        requires_grad=True,
        device=device
    )
    v = torch.rand(
        (batch_size, head_number, in_token_number, head_size),
        requires_grad=True,
        device=device
    )
    pq = torch.rand(
        (1, head_number, 2 * hw + 1, head_size),
        requires_grad=True,
        device=device
    )
    pk = torch.rand(
        (1, head_number, 2 * hw + 1, head_size),
        requires_grad=True,
        device=device
    )
    m = torch.ones(
        (batch_size, in_token_number),
        requires_grad=False,
        device=device,
        dtype=torch.float32
    )
    if in_token_number > 2:
        for i in range(1, batch_size):
            j = random.randint(1, in_token_number-1)
            m[i, j:] = 0.0

    return q, k, v, m, pq, pk, hw


@pytest.fixture
def input_tensors():
    return create_example()


def test_global_forward(input_tensors, atol=1e-6):
    q, k, v, m, _, _, _ = input_tensors
    c_py, p_py, s_py = multi_head_global_attention_function_py(q, k, v, m)
    c, p = fnMultiHeadGlobalAttention(q, k, v, m)
    assert torch.all(torch.isclose(p_py, p, atol=atol))
    assert torch.all(torch.isclose(c_py, c, atol=atol))


def test_local_forward(input_tensors, atol=1e-5):
    q, k, v, m, _, _, hw = input_tensors
    c_py, p_py = multi_head_local_attention_function_py(q, k, v, m, hw)
    c, p = MultiHeadLocalAttention.apply(q, k, v, m, hw)
    pp = torch.zeros_like(p_py)
    nin, nout = k.shape[2], q.shape[2]
    t = 0
    for i in range(nout):
        j_min, j_max = max(0, i - hw), min(i+hw+1, nin)
        pp[..., i, j_min:j_max] = p[..., t:t+(j_max-j_min)]
        t += j_max - j_min
    mt = m.unsqueeze(-1).unsqueeze(1)
    dp = (p_py - pp) * mt
    dc = (c_py - c) * mt

    a = torch.max(torch.abs(dp))
    b = torch.max(torch.abs(dc))
    assert torch.max(torch.abs(dp)) < atol
    assert torch.max(torch.abs(dc)) < atol


def test_local_position_forward(input_tensors, atol=1e-5):
    q, k, v, m, pq, pk, hw = input_tensors
    c_py, p_py = multi_head_local_attention_position_function_py(q, k, v, pq, pk, m, hw)
    c, p = MultiHeadLocalAttention.apply(q, k, v, pq, pk, m, hw)
    pp = torch.zeros_like(p_py)
    nin, nout = k.shape[2], q.shape[2]
    t = 0
    for i in range(nout):
        j_min, j_max = max(0, i - hw), min(i+hw+1, nin)
        pp[..., i, j_min:j_max] = p[..., t:t+(j_max-j_min)]
        t += j_max - j_min
    mt = m.unsqueeze(-1).unsqueeze(1)
    dp = (p_py - pp) * mt
    dc = (c_py - c) * mt

    a = torch.max(torch.abs(dp))
    b = torch.max(torch.abs(dc))
    assert torch.max(torch.abs(dp)) < atol
    assert torch.max(torch.abs(dc)) < atol


def test_global_backward(input_tensors, atol=1e-3):
    q, k, v, m, _, _, _ = input_tensors
    q.grad, k.grad, v.grad = None, None, None
    c_py, p_py, s_py = multi_head_global_attention_function_py(q, k, v, m)
    (c_py.sum() + (p_py * torch.log(p_py + 1e-6)).sum()).backward()
    q_grad_py, k_grad_py, v_grad_py = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None
    c, p = fnMultiHeadGlobalAttention(q, k, v, m)
    (c.sum() + (p * torch.log(p + 1e-6)).sum()).backward()
    q_grad, k_grad, v_grad = q.grad, k.grad, v.grad

    assert torch.all(torch.isclose(q_grad_py, q_grad, atol=atol))
    assert torch.all(torch.isclose(k_grad_py, k_grad, atol=atol))
    assert torch.all(torch.isclose(v_grad_py, v_grad, atol=atol))


def test_local_backward(input_tensors, atol=1e-3):
    q, k, v, m, _, _, hw = input_tensors
    q.grad, k.grad, v.grad = None, None, None
    c_py, p_py = multi_head_local_attention_function_py(q, k, v, m, hw)
    loss = (c_py * m[:, None, :, None]).sum()
    loss.backward()
    q_grad_py, k_grad_py, v_grad_py = q.grad, k.grad, v.grad

    q.grad, k.grad, v.grad = None, None, None
    c, p = MultiHeadLocalAttention.apply(q, k, v, m, torch.tensor(hw))
    loss = (c * m[:, None, :, None]).sum()
    loss.backward()
    q_grad, k_grad, v_grad = q.grad, k.grad, v.grad

    d = torch.max(torch.abs(v_grad_py - v_grad))
    a = torch.max(torch.abs(q_grad_py - q_grad))
    b = torch.max(torch.abs(k_grad_py - k_grad))
    assert torch.all(torch.isclose(q_grad_py, q_grad, atol=atol))
    assert torch.all(torch.isclose(k_grad_py, k_grad, atol=atol))
    assert torch.all(torch.isclose(v_grad_py, v_grad, atol=atol))


def test_local_position_backward(input_tensors, atol=1e-3):
    q, k, v, m, pq, pk, hw = input_tensors
    q.grad, k.grad, v.grad = None, None, None
    c_py, p_py = multi_head_local_attention_position_function_py(q, k, v, pq, pk, m, hw)
    loss = (c_py * m[:, None, :, None]).sum()
    loss.backward()
    q_grad_py, k_grad_py, v_grad_py = q.grad, k.grad, v.grad
    pq_grad_py, pk_grad_py = pq.grad, pk.grad

    q.grad, k.grad, v.grad, pq.grad, pk.grad = None, None, None, None, None
    c, p = MultiHeadLocalAttention.apply(q, k, v, pq, pk, m, hw)
    loss = (c * m[:, None, :, None]).sum()
    loss.backward()
    q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
    pq_grad, pk_grad = pq.grad, pk.grad

    d = torch.max(torch.abs(v_grad_py - v_grad))
    a = torch.max(torch.abs(q_grad_py - q_grad))
    b = torch.max(torch.abs(k_grad_py - k_grad))
    f = torch.max(torch.abs(pq_grad_py - pq_grad))
    e = torch.max(torch.abs(pk_grad_py - pk_grad))
    assert torch.all(torch.isclose(q_grad_py, q_grad, atol=atol))
    assert torch.all(torch.isclose(k_grad_py, k_grad, atol=atol))
    assert torch.all(torch.isclose(v_grad_py, v_grad, atol=atol))
    assert torch.all(torch.isclose(pq_grad_py, pq_grad, atol=atol))
    assert torch.all(torch.isclose(pk_grad_py, pk_grad, atol=atol))


def test_global_speed(num=8):
    forward = 0.0
    backward = 0.0
    forward_py = 0.0
    backward_py = 0.0

    for _ in range(num):
        q, k, v, m, _, _, _ = create_example()

        st = time.time()
        ans_py = multi_head_global_attention_function_py(q, k, v, m)
        forward_py += time.time() - st

        loss = ans_py[0].sum()
        # loss = ans_py[0].sum() + (ans_py[1] * ans_py[2]).sum()

        st = time.time()
        loss.backward()
        backward_py += time.time() - st

        q.grad, k.grad, v.grad, m.grad = None, None, None, None

        st = time.time()
        ans = fnMultiHeadGlobalAttention(q, k, v, m)
        forward += time.time() - st

        loss = ans[0].sum()
        # loss = ans[0].sum() + (ans[1] * ans[2]).sum()

        st = time.time()
        loss.backward()
        backward += time.time() - st

    print()
    print('Python impl')
    print(f'forward: {forward_py / num} sec. backward: {backward_py / num} sec.')
    print('C++ impl')
    print(f'forward: {forward / num} sec. backward: {backward / num} sec.')


def test_local_speed(num=8):
    forward = 0.0
    backward = 0.0
    forward_py = 0.0
    backward_py = 0.0

    for _ in range(num):
        q, k, v, m, _, _, hw = create_example()

        st = time.time()
        ans_py = multi_head_local_attention_function_py(q, k, v, m, hw)
        forward_py += time.time() - st

        loss = ans_py[0].sum()
        # loss = ans_py[0].sum() + (ans_py[1] * ans_py[2]).sum()

        st = time.time()
        loss.backward()
        backward_py += time.time() - st

        q.grad, k.grad, v.grad, m.grad = None, None, None, None

        st = time.time()
        ans = MultiHeadLocalAttention.apply(q, k, v, m, torch.tensor(hw))
        forward += time.time() - st

        loss = ans[0].sum()
        # loss = ans[0].sum() + (ans[1] * ans[2]).sum()

        st = time.time()
        loss.backward()
        backward += time.time() - st

    print()
    print('Python naive impl')
    print(f'forward: {forward_py / num} sec. backward: {backward_py / num} sec.')
    print('Python mine impl')
    print(f'forward: {forward / num} sec. backward: {backward / num} sec.')


def test_local_position_speed(num=8):
    forward = 0.0
    backward = 0.0
    forward_py = 0.0
    backward_py = 0.0

    for _ in range(num):
        q, k, v, m, pq, pk, hw = create_example()

        st = time.time()
        ans_py = multi_head_local_attention_position_function_py(q, k, v, pq, pk, m, hw)
        forward_py += time.time() - st

        loss = ans_py[0].sum()
        # loss = ans_py[0].sum() + (ans_py[1] * ans_py[2]).sum()

        st = time.time()
        loss.backward()
        backward_py += time.time() - st

        q.grad, k.grad, v.grad, m.grad = None, None, None, None

        st = time.time()
        ans = MultiHeadLocalAttention.apply(q, k, v, pq, pk, m, hw)
        forward += time.time() - st

        loss = ans[0].sum()
        # loss = ans[0].sum() + (ans[1] * ans[2]).sum()

        st = time.time()
        loss.backward()
        backward += time.time() - st

    print()
    print('Python naive impl')
    print(f'forward: {forward_py / num} sec. backward: {backward_py / num} sec.')
    print('Python mine impl')
    print(f'forward: {forward / num} sec. backward: {backward / num} sec.')
