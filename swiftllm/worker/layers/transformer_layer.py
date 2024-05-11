import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_forward
from swiftllm.worker.kernels.rotary_emb import rotary_emb_fwd

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weight: LlamaTransformerLayerWeight,
    ):
        self.model_config = model_config
        self.weight = weight
    
    def forward_one_request(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        position_cos: torch.Tensor,	# [num_total_tokens, hidden_size]
        position_sin: torch.Tensor,	# [num_total_tokens, hidden_size]
    ) -> torch.Tensor:
        num_total_tokens = input_embds.size(0)
        # Attn norm
        attn_input = rmsnorm_forward(
            input_embds,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = torch.mm(attn_input, self.weight.q_proj)		# [num_total_tokens, hidden_size]
        k = torch.mm(attn_input, self.weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = torch.mm(attn_input, self.weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(num_total_tokens, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(num_total_tokens, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(num_total_tokens, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_emb_fwd(
            q,
            k,
            position_cos,
            position_sin
        )

        # Attention
        q = q.permute(1, 0, 2)	# [num_q_heads, num_total_tokens, head_dim]
        k = torch.repeat_interleave(k, self.model_config.num_q_heads//self.model_config.num_kv_heads, dim=1).permute(1, 0, 2)	# [num_q_heads, num_total_tokens, head_dim]
        v = torch.repeat_interleave(v, self.model_config.num_q_heads//self.model_config.num_kv_heads, dim=1).permute(1, 0, 2)	# [num_q_heads, num_total_tokens, head_dim]
        mask = torch.triu(
            torch.full(
                (num_total_tokens, num_total_tokens),
                -1e20,
                dtype=torch.float32,
                device=q.device
            ),
        1).unsqueeze(0)	# [1, num_total_tokens, num_total_tokens]
        attn_score = torch.bmm(q.to(torch.float32), k.to(torch.float32).transpose(1, 2))	# [num_q_heads, num_total_tokens, num_total_tokens]
        attn_score = attn_score / torch.sqrt(torch.tensor(self.model_config.head_dim, dtype=torch.float32))
        attn_score = attn_score + mask
        attn_score = torch.softmax(attn_score, dim=-1).to(torch.float16)	# [num_q_heads, num_total_tokens, num_total_tokens]
        attn_output = torch.bmm(attn_score, v)	# [num_q_heads, num_total_tokens, head_dim]
        attn_output = attn_output.transpose(0, 1).reshape(num_total_tokens, -1)	# [num_total_tokens, hidden_size]

        # Attention output GEMM
        attn_output = torch.mm(attn_output, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        # Residual
        attn_output += input_embds

        # FFN norm
        ffn_input = rmsnorm_forward(
            attn_output,
            self.weight.ffn_norm,
            self.model_config.rms_norm_eps
        )
        
        # FFN
        gate = torch.nn.functional.silu(torch.mm(ffn_input, self.weight.gate_proj))
        up = torch.mm(ffn_input, self.weight.up_proj)
        ffn_out = torch.mm(gate * up, self.weight.down_proj)

        # Residual
        ffn_out += attn_output

        return ffn_out
    