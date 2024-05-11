import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState

@triton.jit
def _fwd_paged_attention_phase1(
    mid_o: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks, head_dim], contiguous. num_seq_blocks = ceil(max_seq_len / seq_block_size)
    mid_o_logexpsum: torch.Tensor,	# [num_decoding_seqs, num_q_heads, num_seq_blocks], contiguous
    q: torch.Tensor,    	# [num_decoding_seqs, num_q_heads, head_dim], contiguous
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    softmax_scale: float,
    decoding_seq_lens: torch.Tensor,	# [num_decoding_seqs], contiguous
    seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    num_seq_blocks: int,

    num_layers: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    seq_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    cur_layer: tl.constexpr
):
    # grid shape: [num_decoding_seqs, num_kv_heads, num_seq_blocks]
    my_batch_id = tl.program_id(0)
    my_kv_head_id = tl.program_id(1)
    my_seq_block_id = tl.program_id(2)

    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_start_token_idx = my_seq_block_id * seq_block_size
    my_end_token_idx = tl.minimum(my_start_token_idx + seq_block_size, my_seq_len)
    num_my_heads = num_q_heads // num_kv_heads
    my_start_q_head = my_kv_head_id * num_my_heads
    my_end_q_head = my_start_q_head + num_my_heads

    if my_start_token_idx >= my_end_token_idx:
        return

    offs_q = tl.arange(my_start_q_head, my_end_q_head)[:, None]*head_dim + (tl.arange(0, head_dim)+my_batch_id*num_q_heads*head_dim)[None, :]
    my_q = tl.load(q + offs_q) # [num_q_heads//num_kv_heads, head_dim]

    my_num_blocks = tl.cdiv(
        my_end_token_idx - my_start_token_idx,
        block_size
    )

    max_score = tl.full((num_my_heads,), -float('inf'), tl.float32)
    sum_exp = tl.zeros((num_my_heads,), tl.float32)
    acc = tl.zeros((num_my_heads, head_dim), dtype=tl.float32)

    kv_offset = (cur_layer*num_kv_heads+my_kv_head_id)*block_size*head_dim
    for block_i in range(my_num_blocks):
        block_idx = my_start_token_idx//block_size + block_i
        offs_token = block_i*block_size + my_start_token_idx + tl.arange(0, block_size)
        block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + block_idx)

        offs_k = tl.arange(0, block_size)[:, None] + (tl.arange(0, head_dim)+kv_offset+block_index*num_layers*num_kv_heads*block_size*head_dim)[None, :]
        k_block = tl.load(k_cache + offs_k,
                          mask=offs_token[:, None] < my_seq_len, other=0.0) # [block_size, head_dim]
        attn_score = tl.dot(my_q, k_block.T, out_dtype=tl.float32) # [num_my_heads, block_size]
        attn_score = attn_score * softmax_scale
        attn_score = tl.where(offs_token[None, :] < my_seq_len, attn_score, -float('inf'))
        v_block = tl.load(v_cache + offs_k,
                          mask=offs_token[:, None] < my_seq_len, other=0.0) # [block_size, head_dim])
        
        cur_max_score = tl.max(attn_score, axis=1)  # [num_my_heads]
        new_max_score = tl.maximum(max_score, cur_max_score)
        exp_attn_score = tl.exp(attn_score - new_max_score[:, None])
        old_acc_scale = tl.exp(max_score - new_max_score)

        acc = acc*old_acc_scale[:, None] + tl.dot(exp_attn_score, v_block.to(tl.float32), out_dtype=tl.float32)
        sum_exp = sum_exp*old_acc_scale + tl.sum(exp_attn_score, axis=1)

    offs_mid_o = my_batch_id*num_q_heads*num_seq_blocks*head_dim + my_seq_block_id*head_dim + (tl.arange(my_start_q_head, my_end_q_head)*num_seq_blocks*head_dim)[:, None] + tl.arange(0, head_dim)[None, :]
    tl.store(mid_o + offs_mid_o, acc / sum_exp[:, None])
    offs_mid_o_logexpsum = my_batch_id*num_q_heads*num_seq_blocks + my_seq_block_id + tl.arange(my_start_q_head, my_end_q_head)*num_seq_blocks
    tl.store(mid_o_logexpsum + offs_mid_o_logexpsum, tl.log(sum_exp) + max_score)   # Here tl.log(sum_exp) + max_score = log(sum(e^{a_i}))

def paged_attention_phase1(
    q: torch.Tensor,                    # [num_decoding_seqs, num_q_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int
):
    assert q.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_block_size % engine_config.block_size == 0

    mid_o = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks,
        model_config.head_dim
    ), device=q.device, dtype=torch.float32)
    mid_o_logexpsum = torch.empty((
        infer_state.num_decoding_seqs,
        model_config.num_q_heads,
        infer_state.num_seq_blocks
    ), device=q.device, dtype=torch.float32)

    grid = (infer_state.num_decoding_seqs, model_config.num_kv_heads, infer_state.num_seq_blocks)
    _fwd_paged_attention_phase1[grid](
        mid_o, mid_o_logexpsum,
        q, k_cache, v_cache,
        block_table,
        infer_state.softmax_scale,
        infer_state.decoding_seq_lens,
        infer_state.seq_ids[infer_state.num_prefill_seqs:],
        infer_state.num_seq_blocks,

        model_config.num_layers,
        model_config.num_q_heads,
        model_config.num_kv_heads,
        engine_config.block_size,
        model_config.head_dim,
        infer_state.seq_block_size,
        engine_config.max_blocks_per_seq,
        cur_layer
    )

    return mid_o, mid_o_logexpsum
