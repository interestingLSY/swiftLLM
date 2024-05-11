import time
import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.model import LlamaModel

if __name__ == '__main__':
    model_path = "/data/weights/Llama-3-8B-Instruct-Gradient-1048k/"

    engine_config = EngineConfig(
        model_path = model_path,
        use_dummy = False
    )

    start_time = time.perf_counter()
    model = LlamaModel(engine_config)
    model_creation_time = time.perf_counter()
    print(f"Model creation time: {model_creation_time - start_time:.2f} seconds")
    
    # text = "Life blooms like a flower, far away"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "one two three four five"
    input_ids = tokenizer(text, return_tensors="pt")['input_ids'].cuda().squeeze(0)
    for _ in range(10):
        output_id = model.forward_one_request(input_ids)
        print(tokenizer.decode(output_id), end="", flush=True)
        input_ids = torch.cat([input_ids, torch.tensor([output_id], device="cuda")], dim=0)
    print()