import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_forward

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward_one_request(
        self,
        input_embds: torch.Tensor	# [num_total_tokens, hidden_size]
    ) -> int:
        input_embds = rmsnorm_forward(
            input_embds,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        input_embds = input_embds[-1]	# [hidden_size]
        logits = torch.mm(self.weights.lm_head, input_embds.unsqueeze(1)).squeeze()	# [vocab_size]
        token_index = torch.argmax(logits).item()
        return token_index
    