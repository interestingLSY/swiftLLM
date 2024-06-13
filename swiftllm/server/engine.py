import asyncio
import functools
from typing import AsyncGenerator

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB

from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput
from .scheduler import Scheduler

class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[RawRequest, asyncio.Queue]] = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        self.event_loop = asyncio.get_event_loop()

        print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[Engine] Loading weights...")
        self.model.load_weights()

        print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.engine_config, num_gpu_blocks)

        print("[Engine] Initializing tokenization engine...")
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)

        print("[Engine] Model initialized")
    
    async def add_request_and_streaming(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request
        """
        output_q = asyncio.Queue()
        self.untokenized_raw_requests.append((raw_request, output_q))
        while True:
            step_output = await output_q.get()
            yield step_output
            output_q.task_done()
            if step_output.request.is_finished():
                break

    async def tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [req.prompt for req, _ in cur_untokenized_raw_requests]
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)
            new_requests = [
                Request(raw_request, prompt_token_id, output_q)
                for (raw_request, output_q), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids)
            ]

            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop
    
    async def main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            # Get the next batch from the scheduler
            cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch()
            if not cur_batch and not cur_swap_in and not cur_swap_out:
                # No new batch, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Perform swap in/out
            assert cur_swap_in == []
            assert cur_swap_out == []
            
            # Forward the model
            input_ids = [
                req.prompt_token_ids if req.is_prefill_stage() else [req.output_token_ids[-1]]
                for req in cur_batch
            ]
            seq_ids = [req.request_id for req in cur_batch]
            decoding_seq_lens_list = [
                req.prompt_len + req.get_cur_output_len() + 1
                for req in cur_batch
                if not req.is_prefill_stage()
            ]
            output_tokens = await self._run_on_model_async(
                self.model.forward,
                input_ids,
                seq_ids,
                decoding_seq_lens_list
            )

            # Deal with output tokens
            for req, output_token in zip(cur_batch, output_tokens):
                req.output_token_ids.append(output_token)
                req.output_q.put_nowait(StepOutput(output_token, req))
            
            # Inform the scheduler
            self.scheduler.on_batch_finish(cur_batch)
