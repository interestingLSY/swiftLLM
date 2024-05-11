import dataclasses

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    model_path: str
    use_dummy: bool

    block_size: int
    max_blocks_per_seq: int
    num_blocks: int
    max_seqs_in_block_table: int
