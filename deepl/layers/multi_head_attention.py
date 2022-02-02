import torch
import asyncio
import torch.nn.functional as F


def _build_relative_index(nin, nout, device):
    q_ids = torch.arange(nout, dtype=torch.long, device=device)
    k_ids = torch.arange(nin, dtype=torch.long, device=device)
    rel_pos_ids = q_ids.view(-1, 1) - k_ids.view(1, -1)
    return rel_pos_ids


def fnMultiHeadGlobalAttention(query, key, value, attention_mask, pos_query=None, pos_key=None, hw=None):
    score = torch.matmul(query, key.transpose(-1, 2))
    extended_attention_mask = torch.ones_like(score, dtype=torch.bool)

    if pos_key is not None:
        # context -> position
        score_c2p = torch.matmul(query, pos_key.transpose(-1, -2))
        index_c2p = _build_relative_index(key.shape[-2], query.shape[-2], key.device)
        index_c2p = torch.clamp(-index_c2p + hw, 0, 2 * hw)
        index_c2p = index_c2p.unsqueeze(0).unsqueeze(0)
        index_c2p = index_c2p.expand([query.shape[0], query.shape[1], query.shape[2], index_c2p.shape[-1]])
        score_c2p = torch.gather(score_c2p, dim=-1, index=index_c2p)
        score = score + score_c2p

    if pos_query is not None:
        # position -> context
        score_p2c = torch.matmul(key, pos_query.transpose(-1, -2))
        index_p2c = _build_relative_index(query.shape[-2], key.shape[-2], key.device)
        index_p2c = torch.clamp(-index_p2c + hw, 0, 2 * hw)
        index_p2c = index_p2c.unsqueeze(0).unsqueeze(0)
        index_p2c = index_p2c.expand([key.shape[0], key.shape[1], key.shape[2], index_p2c.shape[-1]])
        score_p2c = torch.gather(score_p2c, dim=-1, index=index_p2c).transpose(-1, -2)
        score = score + score_p2c

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(dtype=torch.bool)
    extended_attention_mask *= attention_mask.unsqueeze(1).unsqueeze(1)

    if hw is not None:
        ones = torch.ones_like(score, dtype=torch.bool)
        ma = torch.triu(ones, diagonal=-hw)
        mb = torch.triu(ones, diagonal=hw + 1)
        extended_attention_mask *= ma
        extended_attention_mask *= ~ mb

    lv = torch.finfo(score.dtype).min
    score = score + lv * (~extended_attention_mask)

    proba = F.softmax(score, dim=-1)
    context = torch.matmul(proba, value)
    return context, proba


async def _calc_batch_attention(idx_ij, query, key, value, score_c2p, score_p2c, mask, hw):
    lv = torch.finfo(key.dtype).min
    proba, context = [], []
    for i, (j_min, j_max) in idx_ij:
        pj_min = j_min - i + hw
        pj_max = hw + j_max - i
        j_p2c_max = min(hw + i, 2 * hw)
        i_p2c_min = max(0, i - hw)

        q = query[..., i].unsqueeze(-2)
        k = key[..., j_min:j_max]
        mask_row = mask[..., j_min:j_max].unsqueeze(1).unsqueeze(1)
        s = torch.matmul(q, k)

        # context 2 position
        s += score_c2p[..., i, pj_min:pj_max].unsqueeze(-2)
        # position 2 context
        a = [i_p2c_min + i for i in range(j_max - j_min)]
        b = [j_p2c_max - i for i in range(j_max - j_min)]
        s += score_p2c[..., a, b].unsqueeze(-2)

        s = s + lv * (~mask_row)
        p = F.softmax(s, dim=-1)
        proba.append(p)
        v = value[..., j_min:j_max]
        c = torch.matmul(v, p.transpose(-1, 2)).transpose(-1, 2)
        context.append(c)
    return proba, context


async def _run_self_attention(b_ij, query, key, value, score_c2p, score_p2c, mask, hw):
    res = await asyncio.gather(*(_calc_batch_attention(idx_ij, query, key, value, score_c2p, score_p2c, mask, hw) for idx_ij in b_ij))
    return res


async def _batch_matmul_spare_dense_tensor(idx_ji, spare_tensor, dense_tensor):
    return torch.cat([
        torch.matmul(
            dense_tensor[..., i_min:i_max],
            spare_tensor[..., jp].unsqueeze(-1)
        ).transpose(-1, 2)
        for _, ((i_min, i_max), jp) in idx_ji
    ], dim=-2)


async def _run_spare_transpose_matmul(b_ji, spare_tensor, dense_tensor):
    res = await asyncio.gather(*(_batch_matmul_spare_dense_tensor(idx_ji, spare_tensor, dense_tensor) for idx_ji in b_ji))
    return res


async def _batch_dkey_calculate(idx_ji, d_score, query, p_query, d_p_key, hw):
    res = []
    for i, ((i_min, i_max), jp) in idx_ji:
        pi_min = i_min - i + hw
        pi_max = hw + i_max - i

        q = query[..., i_min:i_max]
        s = d_score[..., jp].unsqueeze(-1)
        dk = torch.matmul(q, s).transpose(-1, -2)
        pq = p_query[..., pi_min:pi_max]
        dk += torch.matmul(pq, s).transpose(-1, -2)

        sq = torch.sum(s.transpose(-1, -2) * q, dim=0, keepdim=True)
        if pi_min == 0 and pi_max == d_p_key.shape[-1]:
            d_p_key += sq
        else:
            d_p_key[..., pi_min:pi_max] += sq
        res.append(dk)
    res = torch.cat(res, dim=-2)
    return res


async def _run_dkey_calculate(b_ji, d_score, query, p_query, d_p_key, hw):
    return await asyncio.gather(*(_batch_dkey_calculate(idx_ji, d_score, query, p_query, d_p_key, hw) for idx_ji in b_ji))


async def _batch_dscore_dquery_calculate(idx_ij, grad_context, proba, key, value, p_key, d_p_query, hw):
    d_score = []
    d_query = []
    for i, (j_min, j_max), (t_min, t_max) in idx_ij:
        pj_min = j_min - i + hw
        pj_max = hw + j_max - i

        dsi = torch.matmul(
            grad_context[..., i].unsqueeze(-2),
            value[..., j_min:j_max]
        )
        p = proba[..., t_min:t_max].unsqueeze(-2)
        dsi -= torch.sum(dsi * p, dim=-1, keepdim=True)
        dsi *= p
        # calculate d_query
        k = key[..., j_min:j_max]
        dq = torch.matmul(dsi, k.transpose(-1, -2))

        pk = p_key[..., pj_min:pj_max].transpose(-1, -2)
        dq += torch.matmul(dsi, pk)
        d_query.append(dq)

        sk = torch.sum(dsi * k, dim=0, keepdim=True)
        if pj_min == 0 and pj_max == d_p_query.shape[-1]:
            d_p_query += sk
        else:
            d_p_query[..., pj_min:pj_max] += sk
        d_score.append(dsi.squeeze(-2))
    return d_score, d_query


async def _run_dscore_dquery_calculate(b_ij, grad_context, proba, key, value, p_key, d_p_query, hw):
    return await asyncio.gather(*(_batch_dscore_dquery_calculate(idx_ij, grad_context, proba, key, value, p_key, d_p_query, hw) for idx_ij in b_ij))


def __slit_it(it):
    ii = [v[0] for v in it]
    tt = [v[1] for v in it]
    return (min(ii), max(ii) + 1), tt


def _create_index_map(nin, nout, hw):
    idx_ij = []  # вычисляем отображение i -> j
    idx_ji = [[] for _ in range(nin)]  # вычисляем отображение j -> i
    t = 0
    for i in range(nout):
        j_min, j_max = max(0, i - hw), min(i + hw + 1, nin)
        idx_ij.append((i, (j_min, j_max), (t, t + j_max - j_min)))
        for j in range(i - hw, i + hw + 1):
            if 0 <= j < nin:
                idx_ji[j] += [(i, t)]
                t += 1
    idx_ji = [(j, __slit_it(v)) for j, v in enumerate(idx_ji)]
    return idx_ij, idx_ji


class MultiHeadLocalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, p_query, p_key, mask, hw, bs=8):
        query, key, value = query.detach(), key.detach(), value.detach()
        p_query, p_key = p_query.detach(), p_key.detach()
        nin, nout = key.shape[2], query.shape[2]
        idx_ij = [
            (i, (max(0, i - hw), min(i + hw + 1, nin)))
            for i in range(nout)
        ]
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        p_key = p_key.transpose(-1, 2)
        p_query = p_query.transpose(-1, 2)

        score_c2p = torch.matmul(query, p_key)
        score_p2c = torch.matmul(key, p_query)

        query = query.transpose(-1, 2)
        key = key.transpose(-1, 2)
        value = value.transpose(-1, 2)

        b_ij = [idx_ij[i:i+bs] for i in range(0, len(idx_ij), bs)]
        res = asyncio.run(_run_self_attention(
            b_ij, query, key, value, score_c2p, score_p2c, mask, hw
        ))
        proba = [p for pc in res for p in pc[0]]
        context = [c for pc in res for c in pc[1]]
        context = torch.cat(context, dim=-2)

        proba = [p.squeeze(-2) for p in proba]
        proba = torch.cat(proba, dim=-1)

        ctx.save_for_backward(proba, query, key, value, p_query, p_key)
        ctx.hw = hw
        ctx.bs = bs
        return context, proba

    @staticmethod
    def backward(ctx, grad_context, grad_proba):
        proba, query, key, value, p_query, p_key = ctx.saved_tensors
        hw, bs = ctx.hw, ctx.bs
        nin, nout = key.shape[-1], query.shape[-1]
        grad_context = grad_context.transpose(-1, 2)
        idx_ij, idx_ji = _create_index_map(nin, nout, hw)
        b_ji = [idx_ji[i:i + bs] for i in range(0, len(idx_ji), bs)]
        b_ij = [idx_ij[i:i + bs] for i in range(0, len(idx_ij), bs)]

        d_value = asyncio.run(_run_spare_transpose_matmul(
            b_ji, proba, grad_context
        ))
        d_value = torch.cat(d_value, dim=-2)

        d_p_query = torch.zeros_like(p_query)
        sq = asyncio.run(_run_dscore_dquery_calculate(
            b_ij, grad_context, proba, key, value, p_key, d_p_query, hw
        ))
        d_score = [vi for v in sq for vi in v[0]]
        d_query = [vi for v in sq for vi in v[1]]
        d_score = torch.cat(d_score, dim=-1)
        d_query = torch.cat(d_query, dim=-2)

        d_p_key = torch.zeros_like(p_key)
        d_key = asyncio.run(_run_dkey_calculate(
            b_ji, d_score, query, p_query, d_p_key, hw
        ))
        d_key = torch.cat(d_key, dim=-2)

        d_p_query = torch.flip(d_p_query, [-1])
        d_p_key = torch.flip(d_p_key, [-1])

        d_p_query = d_p_query.transpose(-1, -2)
        d_p_key = d_p_key.transpose(-1, -2)
        return d_query, d_key, d_value, d_p_query, d_p_key, None, None, None
