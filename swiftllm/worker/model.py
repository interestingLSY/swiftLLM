import os, json
import itertools

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import cdiv

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

class LlamaModel:
    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        with open(os.path.join(engine_config.model_path, "config.json"), "r", encoding="utf-8") as f:
            self.model_config = json.loads(f.read())
        self.model_config = LlamaModelConfig(self.model_config)

        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            engine_config.model_path,
            engine_config.use_dummy
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize KV Cache
        kvcache_shape = (
            self.engine_config.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_cache = torch.empty(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.empty(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize block manager
        self.block_manager = BlockManager(
            self.engine_config.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)
        torch.cuda.empty_cache()

    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        block_table_cuda = self.block_manager.block_table.to(device="cuda", non_blocking=True)
        input_embds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                block_table_cuda,
                infer_state
            )
        input_embds += residual_buf
        output_tokens = self.post_layer.forward(input_embds, infer_state)
        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids: list[list[int]], # [batch_size, *]
        seq_ids: list[int],     # [batch_size]
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]
        num_prefill_seqs: int,
    ) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server, i.e. to be called
        remotely via Ray.
        """

        flattened_input_ids = list(itertools.chain(*input_ids))
        seq_lengths = [len(seq) for seq in input_ids[:num_prefill_seqs]] + decoding_seq_lens_list

        batch_size = len(input_ids)
        num_tokens = len(flattened_input_ids)

        prefill_seq_lens_list = seq_lengths[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        self.block_manager.allocate_blocks_for_seqs(
            torch.tensor(seq_ids, dtype=torch.int32, device="cpu"),
            torch.tensor(seq_lengths, dtype=torch.int32, device="cpu")
        )

        # Select the seq_block_size
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among the grid, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        seq_block_size = 2048
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2

        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = torch.tensor(seq_ids, dtype=torch.int32, device="cuda"),
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],
        )

        return self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state
        )

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids: list[int]):
        """
        Free the resources of the specified sequences.
        """

        self.block_manager.free_blocks_for_seqs(seq_ids)
