"""
수동 device_map 생성 유틸리티 — BnB 4-bit 대형 모델 로딩용.

accelerate의 `device_map="auto"`는 BF16 원본 크기 기준으로 계산해서
CPU 오프로드를 자동 트리거 → BnB 0.49.2 meta tensor 버그.

이 모듈은 모든 레이어를 정수 GPU ID에 명시 배치하는 dict를 생성한다.

사용법:
    import torch
    from transformers import AutoConfig
    from pipeline.templates.bnb_manual_device_map import build_device_map

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device_map = build_device_map(cfg, n_gpus=torch.cuda.device_count())

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=cfg,
        quantization_config=bnb_config,
        device_map=device_map,  # "auto" 대신 수동
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
"""
from typing import Any


# 모델 아키텍처별 레이어 prefix
# (필요 시 사용자 프로젝트에서 아키텍처 추가)
LAYER_PREFIX_BY_ARCH = {
    "LlamaForCausalLM": "model.layers",
    "MistralForCausalLM": "model.layers",
    "MixtralForCausalLM": "model.layers",
    "GPTNeoXForCausalLM": "gpt_neox.layers",
    "GPT2LMHeadModel": "transformer.h",
}

EMBED_MODULE_BY_ARCH = {
    "LlamaForCausalLM": "model.embed_tokens",
    "MistralForCausalLM": "model.embed_tokens",
    "MixtralForCausalLM": "model.embed_tokens",
    "GPTNeoXForCausalLM": "gpt_neox.embed_in",
    "GPT2LMHeadModel": "transformer.wte",
}

FINAL_MODULES_BY_ARCH = {
    "LlamaForCausalLM": ["model.norm", "lm_head"],
    "MistralForCausalLM": ["model.norm", "lm_head"],
    "MixtralForCausalLM": ["model.norm", "lm_head"],
    "GPTNeoXForCausalLM": ["gpt_neox.final_layer_norm", "embed_out"],
    "GPT2LMHeadModel": ["transformer.ln_f", "lm_head"],
}


def build_device_map(
    config: Any,
    n_gpus: int,
    layer_prefix: str | None = None,
    embed_module: str | None = None,
    final_modules: list[str] | None = None,
) -> dict[str, int]:
    """
    config + GPU 수 → 수동 device_map dict.

    - 모든 값은 정수 GPU ID (0 ~ n_gpus-1)
    - "cpu", "disk" 값 절대 없음 → accelerate offload hook 트리거 안 됨
    - 앞쪽 GPU(embed 포함)에 더 많이 배치 (불균형 완화)

    Args:
        config: HuggingFace 모델 config (num_hidden_layers 필수)
        n_gpus: 사용할 GPU 수
        layer_prefix: 레이어 dict key prefix (None이면 architecture 자동 감지)
        embed_module: embed 모듈 이름
        final_modules: norm/lm_head 등 최종 모듈 목록

    Returns:
        dict[str, int]: device_map, 모든 값이 정수 GPU ID
    """
    # Architecture 감지
    arch = getattr(config, "architectures", None)
    arch_name = arch[0] if arch else "LlamaForCausalLM"

    layer_prefix = layer_prefix or LAYER_PREFIX_BY_ARCH.get(arch_name, "model.layers")
    embed_module = embed_module or EMBED_MODULE_BY_ARCH.get(arch_name, "model.embed_tokens")
    final_modules = final_modules or FINAL_MODULES_BY_ARCH.get(
        arch_name, ["model.norm", "lm_head"]
    )

    n_layers = config.num_hidden_layers
    if n_gpus <= 0:
        raise ValueError("n_gpus must be >= 1")
    if n_gpus > n_layers:
        raise ValueError(f"n_gpus ({n_gpus}) > n_layers ({n_layers})")

    # 레이어 분배: 잔여분을 앞쪽 GPU에 분산
    base = n_layers // n_gpus
    extra = n_layers % n_gpus
    layers_per_gpu = [base + (1 if i < extra else 0) for i in range(n_gpus)]

    # 앞쪽에 embed가 있으므로 배치 반전 (뒷쪽 GPU가 덜 부담)
    layers_per_gpu.reverse()

    device_map: dict[str, int] = {embed_module: 0}
    idx = 0
    for gpu_id, n in enumerate(layers_per_gpu):
        for _ in range(n):
            device_map[f"{layer_prefix}.{idx}"] = gpu_id
            idx += 1

    # final 모듈은 마지막 GPU
    for mod in final_modules:
        device_map[mod] = n_gpus - 1

    return device_map


def summary(device_map: dict[str, int]) -> str:
    """device_map 요약 문자열."""
    from collections import Counter
    gpu_counts = Counter(device_map.values())
    lines = [f"device_map: {len(device_map)} modules across {len(gpu_counts)} GPUs"]
    for gpu_id in sorted(gpu_counts):
        lines.append(f"  GPU {gpu_id}: {gpu_counts[gpu_id]} modules")
    return "\n".join(lines)


if __name__ == "__main__":
    # 예시
    class FakeConfig:
        num_hidden_layers = 32
        architectures = ["LlamaForCausalLM"]

    dm = build_device_map(FakeConfig(), n_gpus=4)
    print(summary(dm))
    print("\nSample entries:")
    for k in list(dm.keys())[:5]:
        print(f"  {k}: {dm[k]}")
    print("  ...")
    for k in list(dm.keys())[-4:]:
        print(f"  {k}: {dm[k]}")
