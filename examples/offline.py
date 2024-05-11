import time
import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.model import LlamaModel

if __name__ == '__main__':
    model_path = "/data/weights/Llama-3-8B-Instruct-Gradient-1048k/"

    engine_config = EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        max_blocks_per_seq = 1024,
        num_blocks = 2048,
        max_seqs_in_block_table = 128
    )

    start_time = time.perf_counter()
    model = LlamaModel(engine_config)
    model_creation_time = time.perf_counter()
    print(f"Model creation time: {model_creation_time - start_time:.2f} seconds")
    
    prompts = [
        "Life blooms like a flower, far away",
        "one two three four five",
        "A B C D E F G H I J K L M N O P Q R S T U V",
        "To be or not to be,",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prompt phase
    input_ids = tokenizer(prompts)['input_ids']
    prompt_phase_outputs = model.forward(
        input_ids,
        list(range(0, len(prompts))),
        [],
        len(prompts)
    )
    print(prompt_phase_outputs)
    print(tokenizer.batch_decode(prompt_phase_outputs, skip_special_tokens=True))
    # for _ in range(10):
    #     output_id = model.forward_one_request(input_ids)
    #     print(tokenizer.decode(output_id), end="", flush=True)
    #     input_ids = torch.cat([input_ids, torch.tensor([output_id], device="cuda")], dim=0)
    # print()