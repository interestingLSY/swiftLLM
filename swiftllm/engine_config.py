import dataclasses
import argparse

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_cpu_blocks: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    max_batch_size: int
    max_tokens_in_batch: int

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.97,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-cpu-blocks",
            type=int,
            default=2048,
            help="Number of CPU blocks",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=4096,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=32768,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=32768,
            help="Maximum number of tokens in a batch",
        )
        