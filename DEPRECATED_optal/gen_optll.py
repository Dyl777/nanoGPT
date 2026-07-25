import pathlib

opt = pathlib.Path(r"C:\Users\AMBE\Desktop\Engineering\nanoGPT\optal")

linear_biasfree = r"""@PL::Python
@Task::LinearBiasFree
@Modulu::Linear
@Technique::Linear::[matmul, addernetmatmul]

// Standard matrix multiply: y = x @ W^T
@Impl::Linear::matmul{
import torch
_x = torch.randn(4, 512)
_w = torch.randn(64, 512)
_out = _x @ _w.t()
@}

// AdderNet L1-distance alternative (no multiply):
// y_o = -sum_i |x_i - w_{o,i}|
@Impl::Linear::addernetmatmul{
import torch
_x = torch.randn(4, 512)
_w = torch.randn(64, 512)
_out = -((_x.unsqueeze(1) - _w.unsqueeze(0)).abs().sum(dim=2))
@}

@Graded::Correctness{
import torch
_ref = _x @ _w.t()
_cos = torch.nn.functional.cosine_similarity(
    _out.reshape(-1), _ref.reshape(-1), dim=0
).item()
@}::endpoints::[cosine_sim=_cos]

@Apply::AllOps
@Run::shuffleAll
@Save::All::endpoints[cosine_sim]
"""

linear_bias = r"""@PL::Python
@Task::LinearBias
@Modulu::Linear
@Technique::Linear::[matmul, addernetmatmul]

// Standard matrix multiply with bias: y = x @ W^T + b
@Impl::Linear::matmul{
import torch
_x = torch.randn(4, 512)
_w = torch.randn(512, 512)
_b = torch.randn(512)
_out = _x @ _w.t() + _b
@}

// AdderNet L1-distance alternative (no multiply):
// y_o = -sum_i |x_i - w_{o,i}| + b
@Impl::Linear::addernetmatmul{
import torch
_x = torch.randn(4, 512)
_w = torch.randn(512, 512)
_b = torch.randn(512)
_out = -((_x.unsqueeze(1) - _w.unsqueeze(0)).abs().sum(dim=2)) + _b
@}

@Graded::Correctness{
import torch
_ref = _x @ _w.t() + _b
_cos = torch.nn.functional.cosine_similarity(
    _out.reshape(-1), _ref.reshape(-1), dim=0
).item()
@}::endpoints::[cosine_sim=_cos]

@Apply::AllOps
@Run::shuffleAll
@Save::All::endpoints[cosine_sim]
"""

attn_scores = r"""@PL::Python
@Task::AttnScores
@Modulu::Attention
@Technique::Attention::[matmul, addernetmatmul]

// Standard scaled dot-product attention scores: S = (q @ k^T) / sqrt(hs)
@Impl::Attention::matmul{
import torch
_b = 2
_h = 4
_t = 8
_hs = 16
_q = torch.randn(_b, _h, _t, _hs)
_k = torch.randn(_b, _h, _t, _hs)
_out = (_q @ _k.transpose(-2, -1)) / (_hs ** 0.5)
@}

// AdderNet alternative (no matmul): replace q@k^T with L1 distance
// S_{i,j} = -sum_d |q_{i,d} - k_{j,d}| / sqrt(hs)
@Impl::Attention::addernetmatmul{
import torch
_b = 2
_h = 4
_t = 8
_hs = 16
_q = torch.randn(_b, _h, _t, _hs)
_k = torch.randn(_b, _h, _t, _hs)
_out = -((_q.unsqueeze(-2) - _k.unsqueeze(-3)).abs().sum(dim=-1)) / (_hs ** 0.5)
@}

@Graded::Correctness{
import torch
_ref = (_q @ _k.transpose(-2, -1)) / (_hs ** 0.5)
_cos = torch.nn.functional.cosine_similarity(
    _out.reshape(-1), _ref.reshape(-1), dim=0
).item()
@}::endpoints::[cosine_sim=_cos]

@Apply::AllOps
@Run::shuffleAll
@Save::All::endpoints[cosine_sim]
"""

attn_context = r"""@PL::Python
@Task::AttnContext
@Modulu::Attention
@Technique::Attention::[matmul, addernetmatmul]

// Standard context aggregation: C = attn @ v
@Impl::Attention::matmul{
import torch
_b = 2
_h = 4
_t = 8
_hs = 16
_attn = torch.randn(_b, _h, _t, _t)
_v = torch.randn(_b, _h, _t, _hs)
_out = _attn @ _v
@}

// AdderNet alternative (no matmul): replace attn@v with L1 distance
// C_{i,d} = -sum_j |attn_{i,j} - v_{j,d}|
@Impl::Attention::addernetmatmul{
import torch
_b = 2
_h = 4
_t = 8
_hs = 16
_attn = torch.randn(_b, _h, _t, _t)
_v = torch.randn(_b, _h, _t, _hs)
_out = -((_attn.unsqueeze(-1) - _v.unsqueeze(-2)).abs().sum(dim=-2))
@}

@Graded::Correctness{
import torch
_ref = _attn @ _v
_cos = torch.nn.functional.cosine_similarity(
    _out.reshape(-1), _ref.reshape(-1), dim=0
).item()
@}::endpoints::[cosine_sim=_cos]

@Apply::AllOps
@Run::shuffleAll
@Save::All::endpoints[cosine_sim]
"""

files = {
    "linear_biasfree": linear_biasfree,
    "linear_bias": linear_bias,
    "attn_scores": attn_scores,
    "attn_context": attn_context,
}
for name, content in files.items():
    p = opt / (name + ".optll")
    p.write_text(content, encoding="utf-8")  # no BOM
    print(f"{name}.optll  bom={p.read_bytes()[:3] == b'\xef\xbb\xbf'}  bytes={len(content)}")
