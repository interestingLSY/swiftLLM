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
        # self.qkv_proj = torch.cat((self.q_proj, self.k_proj, self.v_proj), dim=0).transpose(0, 1).contiguous()
        # del self.q_proj, self.k_proj, self.v_proj
        self.q_proj = self.q_proj.transpose(0, 1).contiguous()
        self.k_proj = self.k_proj.transpose(0, 1).contiguous()
        self.v_proj = self.v_proj.transpose(0, 1).contiguous()
        self.o_proj = self.o_proj.transpose(0, 1).contiguous()
        self.up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).transpose(0, 1).contiguous()
        del self.up_proj, self.gate_proj
        self.down_proj = self.down_proj.transpose(0, 1).contiguous()


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
        self.lm_head = self.lm_head.transpose(0, 1).contiguous()
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
        num_safetensor_files = len([name for name in os.listdir(model_path) if name.endswith(".safetensors")])
        if num_safetensor_files > 0:
            # Use Safetensors
            # Here we assume the weight is stored in multiple files
            safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
            with open(safetensor_index_path, "r", encoding="utf-8") as f:
                safetensor_index = json.load(f)["weight_map"]
            def weight_getter_real(item: RegisteredWeightItem):
                file_name = safetensor_index[item.key]
                file_path = os.path.join(model_path, file_name)
                with safetensors.safe_open(file_path, framework="pt", device="cuda") as f:
                    tensor = f.get_tensor(item.key)
                return tensor.to(item.dtype)
            getter = weight_getter_real
        else:
            # Use PyTorch
            # Here we assume the weight is stored in multiple files
            raise NotImplementedError("Loading weights from PyTorch is not supported yet")

    weight = LlamaWeight(model_config, dtype)
    weight.load_weights(getter)
    return weight
