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
    outputs = []

    # Prompt phase
    input_ids = tokenizer(prompts)['input_ids']
    prompt_phase_outputs = model.forward(
        input_ids,
        list(range(0, len(prompts))),
        [],
        len(prompts)
    )
    print(tokenizer.batch_decode(prompt_phase_outputs, skip_special_tokens=True))
    outputs.append(prompt_phase_outputs)

    seq_lens = [len(x) for x in input_ids]
    last_round_outputs = prompt_phase_outputs
    for _ in range(20):
        for i, _ in enumerate(prompts):
            seq_lens[i] += 1
        start_time = time.perf_counter()
        last_round_outputs = model.forward(
            [[x] for x in last_round_outputs],
            list(range(0, len(prompts))),
            seq_lens,
            0
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print(f"Forward time (decoding): {(end_time - start_time)*1e3:.2f} ms")
        print(tokenizer.batch_decode(last_round_outputs, skip_special_tokens=True))
        outputs.append(last_round_outputs)
    
    for i, prompt in enumerate(prompts):
        output_tokens = [x[i] for x in outputs]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"{prompt} | {output_text}")
