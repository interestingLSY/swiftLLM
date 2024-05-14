import asyncio
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.server.engine import Engine
from swiftllm.server.structs import RawRequest

print_lock = asyncio.Lock()

async def send_request_and_wait(engine: Engine, tokenizer: AutoTokenizer, prompt: str, output_len: int):
    output_q = engine.add_raw_request(RawRequest(prompt, output_len))
    output_token_ids = []
    async for step_output in engine.streaming_output(output_q):
        output_token_ids.append(step_output.token_id)
    async with print_lock:
        print("---------------------------------")
        print(f"Prompt: {prompt}")
        print(f"Output: {tokenizer.decode(output_token_ids)}")

async def main():
    model_path = "/data/weights/Llama-3-8B-Instruct-Gradient-1048k/"

    engine_config = EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.99,
        num_cpu_blocks = 1024,
        max_seqs_in_block_table = 128,
        max_blocks_per_seq = 3072,

        max_batch_size = 4,
        max_tokens_in_batch = 1024
    )

    prompt_and_output_lens = [
        ("Life blooms like a flower, far away", 10),
        ("one two three four five", 50),
        ("A B C D E F G H I J K L M N O P Q R S T U V", 5),
        ("To be or not to be,", 15),
    ]

    engine = Engine(engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    await engine.initialize()
    asyncio.create_task(engine.main_event_loop())
    asyncio.create_task(engine.tokenize_raw_request_event_loop())

    tasks = []
    for prompt, output_len in prompt_and_output_lens:
        task = asyncio.create_task(send_request_and_wait(engine, tokenizer, prompt, output_len))
        tasks.append(task)
        await asyncio.sleep(0.2)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
