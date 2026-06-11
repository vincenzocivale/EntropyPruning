from peft import LoraConfig
from peft.tuners.lora import LoraModel

# Shared LoRA target modules / hyperparameters: must match between Phase 1
# (GenericLoRAClassifier) and Phase 3 (GenericLoRAWithForecasterPruning) so that
# Phase 3 can warm-start its backbone+head from a Phase-1 checkpoint via
# load_state_dict.
LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]


def wrap_lora(backbone, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1) -> LoraModel:
    """Wrap a raw timm backbone with LoRA adapters on qkv/proj/fc1/fc2 of every block."""
    config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=lora_dropout, bias="none",
    )
    return LoraModel(backbone, config, adapter_name="default")
