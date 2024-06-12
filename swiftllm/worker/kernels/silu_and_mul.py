import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_silu_and_mul(
    x: torch.Tensor,	# [num_tokens, 2*ffn_inter_dim]. Result will be stored at input[:, :ffn_inter_dim]
    ffn_inter_dim: tl.constexpr,
    block_size: tl.constexpr
):
    # grid shape: [num_tokens, ffn_inter_dim / block_size]
    # require ffn_inter_dim % block_size == 0
    my_token_id = tl.program_id(0).to(tl.int64)
    my_block_id = tl.program_id(1)

    offs = my_token_id*(2*ffn_inter_dim) + my_block_id*block_size + tl.arange(0, block_size)
    gate = tl.load(x + (offs+ffn_inter_dim))
    gate = gate.to(tl.float32)
    gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(tl.float16)
    up = tl.load(x + offs)
    down = up * gate
    tl.store(x + offs, down)

def silu_and_mul_inplace(
    x: torch.Tensor # [num_tokens, 2*ffn_inter_dim]
):
    assert x.is_contiguous()
    num_tokens = x.shape[0]
    ffn_inter_dim = x.shape[1] // 2

    block_size = 256
    assert ffn_inter_dim % block_size == 0
    _fwd_silu_and_mul[(num_tokens, ffn_inter_dim//block_size)](x, ffn_inter_dim, block_size)
