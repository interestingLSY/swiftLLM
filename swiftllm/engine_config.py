import dataclasses

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    model_path: str
    use_dummy: bool = False

