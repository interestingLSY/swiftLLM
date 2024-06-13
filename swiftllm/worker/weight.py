import json
import os
import dataclasses
import torch
import safetensors

from swiftllm.model_config import LlamaModelConfig

@dataclasses.dataclass
class RegisteredWeightItem:
    attr_name: str
    key: str
    shape: tuple
    dtype: torch.dtype

class WeightBase:
    """
    The base class of all weight classes (i.e. LlamaTransformerLayerWeight or LlamaWeight)

    During weight initialization, each concrete weight class should first register
    all weight items. Each weight item has its own attribute name, key, shape, and dtype.

    During weight loading, RegisterWeightItem will be passed to the weight getter
    function, which should return the corresponding weight value (real/dummy).
    """

    def __init__(self):
        self.registered_weights = []

    def register_weight(self, item: RegisteredWeightItem):
        self.registered_weights.append(item)

    def _post_process_after_load(self, getter: callable):
        """
        This function is called after loading weights (real/dummy).
        Defined in each concrete weight class, called by load_weights().
        """
        raise NotImplementedError()
    
    def load_weights(self, getter: callable):
        """
        Load weights
        """
        for item in self.registered_weights:
            weight_value = getter(item)
            assert weight_value is not None, f"getter() returned None for {item.key} ({item})"
            assert isinstance(weight_value, torch.Tensor), f"Weight {item.key} is not a tensor"
            assert weight_value.shape == item.shape, f"Shape of weight {item.key} does not match"
            assert weight_value.device.type == "cuda", f"Weight {item.key} is not on GPU"
            setattr(self, item.attr_name, weight_value.to(item.dtype))
        self._post_process_after_load(getter)


class LlamaTransformerLayerWeight(WeightBase):
    """
    Class stores the weights of one transformer layer (transformer block) in Llama model.
    """

    def __init__(
        self,
        layer_id: int,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "attn_norm",
            f"model.layers.{self.layer_id}.input_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "q_proj",
            f"model.layers.{self.layer_id}.self_attn.q_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "k_proj",
            f"model.layers.{self.layer_id}.self_attn.k_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "v_proj",
            f"model.layers.{self.layer_id}.self_attn.v_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "o_proj",
            f"model.layers.{self.layer_id}.self_attn.o_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype
        ))

        self.register_weight(RegisteredWeightItem(
            "ffn_norm",
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "up_proj",
            f"model.layers.{self.layer_id}.mlp.up_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "gate_proj",
            f"model.layers.{self.layer_id}.mlp.gate_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "down_proj",
            f"model.layers.{self.layer_id}.mlp.down_proj.weight",
            (self.model_config.hidden_size, self.model_config.ffn_inter_dim),
            self.dtype
        ))

    def _post_process_after_load(self, getter: callable):
        # pylint: disable=no-member
        # self.qkv_proj = torch.cat((self.q_proj, self.k_proj, self.v_proj), dim=0).contiguous()
        # del self.q_proj, self.k_proj, self.v_proj
        self.up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).contiguous()
        del self.up_proj, self.gate_proj


class LlamaWeight(WeightBase):
    def __init__(
        self,
        model_config: LlamaModelConfig,
        dtype: torch.dtype
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype

        self.register_weight(RegisteredWeightItem(
            "wte",
            "model.embed_tokens.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "lm_head",
            "lm_head.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "final_norm",
            "model.norm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))

        self.layers: list[LlamaTransformerLayerWeight] = []
        for i in range(self.model_config.num_layers):
            layer = LlamaTransformerLayerWeight(i, self.model_config, self.dtype)
            self.layers.append(layer)

    def _post_process_after_load(self, getter: callable):
        for layer in self.layers:
            layer.load_weights(getter)


def load_weights(
    model_config: LlamaModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False
) -> LlamaWeight:
    """
    Load weights from a given path
    """
    if use_dummy:
        def weight_getter_dummy(item: RegisteredWeightItem):
            return torch.empty(item.shape, dtype=item.dtype, device="cuda").uniform_(-0.001, 0.001)
        getter = weight_getter_dummy
    else:
        safetensor_files = [name for name in os.listdir(model_path) if name.endswith(".safetensors")]
        if len(safetensor_files) > 0:
            # Use Safetensors
            safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(safetensor_index_path):
                # The weight is stored in multiple files
                f = open(safetensor_index_path, "r", encoding="utf-8")
                safetensor_index = json.load(f)["weight_map"]
                safetensor_filename = None
            else:
                # The weight is stored in a single file
                assert len(safetensor_files) == 1, "model.safetensors.index.json not found, but there are multiple .safetensors files"
                safetensor_index = None
                safetensor_filename = safetensor_files[0]

            def weight_getter_real(item: RegisteredWeightItem):
                file_name = safetensor_index[item.key] if safetensor_index is not None else safetensor_filename
                file_path = os.path.join(model_path, file_name)
                # For safetensor files, since "opening" it is cheap, we open it every time
                with safetensors.safe_open(file_path, framework="pt", device="cuda") as f:
                    tensor = f.get_tensor(item.key)
                return tensor.to(item.dtype)
            getter = weight_getter_real

        else:
            # Use PyTorch
            pytorch_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            if os.path.exists(pytorch_index_path):
                # The weight is stored in multiple files
                f = open(pytorch_index_path, "r", encoding="utf-8")
                pytorch_index = json.load(f)["weight_map"]
                pytorch_filename = None
            else:
                # The weight is stored in a single file
                pytorch_index = None
                pytorch_filename = "pytorch_model.bin"
            
            # For PyTorch files, since "opening" it is slow (due to deserialization),
            # we open it only once and then store the opened files in a dictionary.
            # We add `mmap=True` to avoid loading the entire file into memory.
            opened_files = {}
            def weight_getter_real(item: RegisteredWeightItem):
                file_name = pytorch_index[item.key] if pytorch_index is not None else pytorch_filename
                file_path = os.path.join(model_path, file_name)
                if file_path not in opened_files:
                    opened_files[file_path] = torch.load(file_path, map_location="cuda", mmap=True)
                file = opened_files[file_path]
                return file[item.key].to(item.dtype)
            getter = weight_getter_real

    weight = LlamaWeight(model_config, dtype)
    weight.load_weights(getter)
    return weight
