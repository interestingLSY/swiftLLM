import time
import argparse
from transformers import AutoTokenizer

import swiftllm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        required=True
    )
    model_path = parser.parse_args().model_path

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.99,
        num_cpu_blocks = 0,
        max_seqs_in_block_table = 128,
        max_blocks_per_seq = 2048,

        # The following are not used in the offline example
        max_batch_size = 16,
        max_tokens_in_batch = 2048*16
    )

    start_time = time.perf_counter()
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    num_blocks = model.profile_num_blocks()
    print("Number of blocks:", num_blocks)
    model.init_kvcache_and_swap(num_blocks)
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
        []
    )
    # print(tokenizer.batch_decode(prompt_phase_outputs, skip_special_tokens=True))
    outputs.append(prompt_phase_outputs)

    seq_lens = [len(x) for x in input_ids]
    last_round_outputs = prompt_phase_outputs
    for _ in range(20):
        for i, _ in enumerate(prompts):
            seq_lens[i] += 1
        last_round_outputs = model.forward(
            [[x] for x in last_round_outputs],
            list(range(0, len(prompts))),
            seq_lens
        )
        # print(tokenizer.batch_decode(last_round_outputs, skip_special_tokens=True))
        outputs.append(last_round_outputs)
    
    for i, prompt in enumerate(prompts):
        output_tokens = [x[i] for x in outputs]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"{prompt}|{output_text}")
