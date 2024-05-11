class LlamaModelConfig:
    """
    The configuration of a LLaMA model (including LLaMA 1/2/3).
    """
    
    def __init__(
        self,
        model_config: dict
    ):
        """
        Initialize a LLaMA model configuration from a dict, which should be generated
        from a huggingface transformers config.json file.
        """
        
        assert model_config["model_type"] == "llama"
        self.num_layers = model_config["num_hidden_layers"]
        self.num_q_heads = model_config["num_attention_heads"]
        self.num_kv_heads = model_config["num_key_value_heads"]
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.rope_scaling = model_config.get("rope_scaling", 1.0)
        self.rope_theta = model_config.get("rope_theta", 10000)
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        assert model_config["hidden_act"] == "silu"
        