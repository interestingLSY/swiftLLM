import os, json

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer

class LlamaModel:
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        self.engine_config = engine_config

        # Load model config
        with open(os.path.join(engine_config.model_path, "config.json"), "r", encoding="utf-8") as f:
            self.model_config = json.loads(f.read())
        self.model_config = LlamaModelConfig(self.model_config)

        self.weight = load_weights(
            self.model_config,
            torch.float16,
            engine_config.model_path,
            engine_config.use_dummy
        )

        self._init_to_get_rotary()

        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.weight.layers[layer_id],
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)
    
    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cpu", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 1024 * 128, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()

    def forward_one_request(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
    ) -> int:
        input_embds = self.pre_layer.forward_one_request(input_ids)
        for layer in self.transformer_layers:
            input_embds = layer.forward_one_request(
                input_embds,
                self._cos_cached[:len(input_ids)],
                self._sin_cached[:len(input_ids)]
            )
        output_id = self.post_layer.forward_one_request(input_embds)
        return output_id
    