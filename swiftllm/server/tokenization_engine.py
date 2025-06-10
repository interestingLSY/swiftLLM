import ray
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig

@ray.remote
class TokenizationEngine:
    def __init__(self, engine_config: EngineConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(engine_config.model_path)

    def batched_tokenize(self, prompts: list[str]) -> list[list[int]]:
        prompt_token_ids = self.tokenizer(prompts, return_attention_mask=False)['input_ids']
        return prompt_token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)