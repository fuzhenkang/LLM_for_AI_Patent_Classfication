"""Model defaults for LLM next-token patent classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModelConfig:
    model_key: str
    base_model: str
    lora_target_modules: str
    max_len: int
    batch_size: int
    lr: float
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    recommend_quantization: bool = False


MODEL_CONFIGS: dict[str, LLMModelConfig] = {
    "llama": LLMModelConfig(
        model_key="llama",
        base_model="meta-llama/Llama-3.2-1B",
        lora_target_modules="q_proj,v_proj",
        max_len=256,
        batch_size=2,
        lr=2e-5,
    ),
    "qwen": LLMModelConfig(
        model_key="qwen",
        base_model="Qwen/Qwen2.5-0.5B",
        lora_target_modules="q_proj,v_proj",
        max_len=256,
        batch_size=2,
        lr=2e-5,
    ),
    "glm": LLMModelConfig(
        model_key="glm",
        base_model="THUDM/glm-4-9b-chat",
        lora_target_modules="query_key_value",
        max_len=256,
        batch_size=1,
        lr=2e-5,
        trust_remote_code=True,
        recommend_quantization=True,
    ),
    "mistral": LLMModelConfig(
        model_key="mistral",
        base_model="mistralai/Mistral-7B-v0.1",
        lora_target_modules="q_proj,v_proj",
        max_len=256,
        batch_size=1,
        lr=2e-5,
        recommend_quantization=True,
    ),
    "baichuan": LLMModelConfig(
        model_key="baichuan",
        base_model="baichuan-inc/Baichuan2-7B-Base",
        lora_target_modules="W_pack",
        max_len=256,
        batch_size=1,
        lr=2e-5,
        trust_remote_code=True,
        recommend_quantization=True,
    ),
}


def get_llm_model_config(model_key: str) -> LLMModelConfig:
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model_key: {model_key}. Available: {available}")
    return MODEL_CONFIGS[model_key]
