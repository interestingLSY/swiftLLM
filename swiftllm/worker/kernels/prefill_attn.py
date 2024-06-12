import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState

@triton.jit
def _fwd_prefill_attention(
    o: torch.Tensor,	# [num_prefill_tokens, num_q_heads, head_dim]
    q: torch.Tensor,	# [num_prefill_tokens, num_q_heads, head_dim]
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim]
    softmax_scale: float,

    prefill_seq_start_locs: torch.Tensor,	# [num_prefill_seqs+1]
    prefill_seq_lens: torch.Tensor,	# [num_prefill_seqs]

    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gpa_group_size: tl.constexpr,	# = num_q_heads // num_kv_heads
    head_dim: tl.constexpr,

    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # grid shape: [num_prefill_seqs, num_q_heads, cdiv(max_prefill_len, BLOCK_Q)]
    # Require: BLOCK_Q % BLOCK_K == 0
    my_batch_id = tl.program_id(0)
    my_q_head = tl.program_id(1)
    my_q_block = tl.program_id(2)
    my_kv_head = my_q_head // gpa_group_size

    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    if my_q_block * BLOCK_Q >= my_seq_len:
        return
    my_q_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    
    q += (my_q_start_loc*num_q_heads+my_q_head)*head_dim
    k += (my_q_start_loc*num_kv_heads+my_kv_head)*head_dim
    v += (my_q_start_loc*num_kv_heads+my_kv_head)*head_dim
    o += (my_q_start_loc*num_q_heads+my_q_head)*head_dim

    range_my_q = my_q_block*BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_my_q = range_my_q[:, None]*(num_q_heads*head_dim) + tl.arange(0, head_dim)[None, :]
    my_q = tl.load(q + offs_my_q, mask = range_my_q[:, None] < my_seq_len, cache_modifier=".cg") # [BLOCK_Q, head_dim]

    k_ptrs = k + (tl.arange(0, BLOCK_K))[None, :]*(num_kv_heads*head_dim) + tl.arange(0, head_dim)[:, None]
    v_ptrs = v + (tl.arange(0, BLOCK_K))[:, None]*(num_kv_heads*head_dim) + tl.arange(0, head_dim)[None, :]

    m_i = tl.full([BLOCK_Q], value=float("-1e20"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)
    
    # Calculate non-diagonal attention
    for k_block_start in range(0, my_q_block*BLOCK_Q, BLOCK_K):
        k_block_start = tl.multiple_of(k_block_start, BLOCK_K)
        # Here masking is unnecessary
        cur_k = tl.load(k_ptrs + k_block_start*(num_kv_heads*head_dim), cache_modifier=".cg") # [head_dim, BLOCK_K]
        qk = tl.dot(my_q, cur_k, out_dtype=tl.float32) * softmax_scale # [BLOCK_Q, BLOCK_K]
        cur_k = None
        cur_v = tl.load(v_ptrs + k_block_start*(num_kv_heads*head_dim), cache_modifier=".cg") # [BLOCK_K, head_dim]
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        exp_qk = tl.math.exp2(qk - m_i_new[:, None])

        m_i = m_i_new
        l_i = l_i*alpha + tl.sum(exp_qk, 1)
        acc = acc*alpha[:, None] + tl.dot(exp_qk.to(tl.float16), cur_v)
    
    # Calculate the diagonal attention
    for k_block_start in range(my_q_block*BLOCK_Q, (my_q_block+1)*BLOCK_Q, BLOCK_K):
        k_block_start = tl.multiple_of(k_block_start, BLOCK_K)
        cur_k = tl.load(k_ptrs + k_block_start*(num_kv_heads*head_dim),
                        mask = (k_block_start + tl.arange(0, BLOCK_K))[None, :] < my_seq_len,
                        cache_modifier=".cg")    # [head_dim, BLOCK_K]
        qk = tl.dot(my_q, cur_k, out_dtype=tl.float32) * softmax_scale  # [BLOCK_Q, BLOCK_K]
        cur_k = None
        cur_v = tl.load(v_ptrs + k_block_start*(num_kv_heads*head_dim),
                        mask = (k_block_start + tl.arange(0, BLOCK_K))[:, None] < my_seq_len,
                        cache_modifier=".cg")    # [BLOCK_K, head_dim]
        
        qk = tl.where(
            ((k_block_start + tl.arange(0, BLOCK_K)) < my_seq_len) & 
            (range_my_q[:, None] >= (k_block_start + tl.arange(0, BLOCK_K))[None, :]),
            qk,
            float("-1e20")
        )

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        exp_qk = tl.math.exp2(qk - m_i_new[:, None])

        m_i = m_i_new
        l_i = l_i*alpha + tl.sum(exp_qk, 1)
        acc = acc*alpha[:, None] + tl.dot(exp_qk.to(tl.float16), cur_v)

    tl.store(o + offs_my_q, acc / l_i[:, None], mask=range_my_q[:, None] < my_seq_len, cache_modifier=".cg")

def prefill_attention(
    q: torch.Tensor,    # [num_prefill_tokens, num_q_heads, head_dim]
    k: torch.Tensor,    # [num_prefill_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,    # [num_prefill_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,    # [num_prefill_tokens, num_q_heads, head_dim]
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
):
    is_rtx4090 = '4090' in torch.cuda.get_device_name(0)
    BLOCK_Q = 128 if not is_rtx4090 else 128
    BLOCK_K = 128 if not is_rtx4090 else 64

    # Here we reduce BLOCK_Q and BLOCK_K, since that when max_prefill_len is
    # small, large block size introduces unnecessary computation when computing
    # the attention score.
    # note: We restrict BLOCK_Q and BLOCK_K >= 16 due to a limitation proposed by tl.dot
    BLOCK_Q = min(BLOCK_Q, triton.next_power_of_2(max(infer_state.max_prefill_len, 16)))
    BLOCK_K = min(BLOCK_K, triton.next_power_of_2(max(infer_state.max_prefill_len, 16)))

    # Please refer to `paged_attn.py` for the reason of multiplying softmax_scale
    # by log2(e)
    softmax_scale2 = infer_state.softmax_scale * 1.442695040888963

    assert BLOCK_Q % BLOCK_K == 0
    grid = (infer_state.num_prefill_seqs, model_config.num_q_heads, triton.cdiv(infer_state.max_prefill_len, BLOCK_Q))
    num_warps = 8
    _fwd_prefill_attention[grid](
        o, q, k, v,
        softmax_scale2,
        infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
        model_config.num_q_heads, model_config.num_kv_heads,
        model_config.num_q_heads // model_config.num_kv_heads,
        model_config.head_dim,
        BLOCK_Q, BLOCK_K,
        num_warps=num_warps,
        num_stages=3
    )
