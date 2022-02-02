/*
score tensor represent as #token x 2w + 1, where w is half width of local attention
*/
#include <vector>
#include <iostream>
#include <torch/extension.h>

auto LOWEST_FLOAT32_VALUE = std::numeric_limits<float>::lowest();

/* Global multi-head attention algorithm Forward and Backward functions
Args:
    query: (#batchs x #heads x #tokens_out x head size) - shape
    key: (#batchs x #heads x #tokens_in x head size) - shape
    value: (#batchs x #heads x #tokens_in x head size) - shape
    extended_attention_mask: (#batchs x #heads x #tokens_out x #tokens_in)

    #tokens_in == #tokens_out for self attention
    #tokens_in != #tokens_out for cross attention
 */
std::vector<at::Tensor> multi_head_global_attention_forward(
    const torch::Tensor query,
    const torch::Tensor key,
    const torch::Tensor value,
    const torch::Tensor extended_attention_mask
) {
    auto score = torch::matmul(query, key.transpose(-1, -2));
    if (!torch::all(extended_attention_mask).item<bool>())
        score.masked_fill_(~extended_attention_mask, LOWEST_FLOAT32_VALUE);
    auto proba = torch::softmax(score, /*dim=*/-1);
    auto context = torch::matmul(proba, value);
    return {
        context,
        proba,
        score
    };
}

std::vector<at::Tensor> multi_head_global_attention_backward(
    const torch::Tensor grad_context,
    const torch::Tensor grad_proba,
    const torch::Tensor grad_score,
    const torch::Tensor proba,
    const torch::Tensor query,
    const torch::Tensor key,
    const torch::Tensor value
) {
    auto d_value = torch::matmul(proba.transpose(-1, -2), grad_context);
    // auto cv = torch::matmul(grad_context, value.transpose(-1, -2));
    // auto acv = torch::sum(proba * cv, /*dim=*/-1, /*keepdim=*/true);
    // auto pdcv = proba * (cv - acv);
    auto d_score = torch::matmul(grad_context, value.transpose(-1, -2));
    d_score -= torch::sum(proba * d_score, /*dim=*/-1, /*keepdim=*/true);
    d_score *= proba;
    auto d_key = torch::matmul(d_score.transpose(-1, -2), query);
    auto d_query = torch::matmul(d_score, key);
    if (grad_proba.any().item<bool>()) {
        auto aa = grad_proba * proba;
        auto saa = torch::sum(aa, /*dim=*/-1, /*keepdim=*/true);
        d_query += torch::matmul(aa, key);
        d_query -= saa * torch::matmul(proba, key);
        d_key += torch::matmul(aa.transpose(-1, -2) - (saa * proba).transpose(-1, -2), query);
    }
    if (grad_score.any().item<bool>()) {
        d_query += torch::matmul(grad_score, key);
        d_key += torch::matmul(grad_score.transpose(-1, -2), query);
    }
    return {
        d_query,
        d_key,
        d_value
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("global_forward", &multi_head_global_attention_forward, "multi head global attention forward");
  m.def("global_backward", &multi_head_global_attention_backward, "multi head global attention backward");
}