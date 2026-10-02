"""
대형 모델 BnB 4-bit 로딩 레퍼런스 템플릿.

CHANGELOG.md의 모든 크래시 사례를 반영한 안전 로딩 패턴.

Usage:
    # 스크립트 최상단
    from pipeline.templates.env_setup import setup_env
    setup_env("/workspace/.cache_hf")

    # 그 다음 torch 관련 import
    from pipeline.templates import large_model_loader_guard as loader

    model, tokenizer = loader.load_large_model_4bit(
        model_path="/workspace/.cache_hf/models--ORG--MODEL/snapshots/XXX",
    )
"""
import os
import sys
import time
import subprocess


def _install_deps_if_needed() -> None:
    """필수 패키지 설치 (pod 재시작 시 재실행)."""
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "transformers==4.48.3",
            "accelerate==1.13.0",
            "bitsandbytes==0.49.2",
            "scikit-learn",
        ],
        check=True,
    )


def _clear_module_cache(hf_home: str, org_name: str) -> None:
    """transformers custom modeling 캐시 정리 (버전 conflict 방지)."""
    import shutil
    cache_path = os.path.join(hf_home, "modules", "transformers_modules", org_name)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print(f"[cache] removed {cache_path}")


def _check_memory_budget(
    n_params_b: float, n_gpus: int, gpu_vram_gb: float
) -> None:
    """BnB 4-bit 메모리 예산 사전 검증."""
    model_size_gb = n_params_b * 0.5 * 1.15  # 4-bit + 15% overhead
    gpu_budget_gb = n_gpus * gpu_vram_gb * 0.85  # 15% workspace

    if model_size_gb > gpu_budget_gb:
        required_gpus = int(
            (n_params_b * 0.5 * 1.15) / (gpu_vram_gb * 0.85)
        ) + 1
        raise RuntimeError(
            f"VRAM 부족: 모델 {model_size_gb:.0f}GB > 예산 {gpu_budget_gb:.0f}GB. "
            f"필요 GPU: {required_gpus}x {gpu_vram_gb:.0f}GB"
        )

    print(
        f"[OK] 메모리 예산: 모델 {model_size_gb:.0f}GB, "
        f"GPU 예산 {gpu_budget_gb:.0f}GB, "
        f"여유 {gpu_budget_gb - model_size_gb:.0f}GB"
    )


def load_large_model_4bit(
    model_path: str,
    install_deps: bool = True,
    clear_module_cache_for: str | None = None,
    n_params_b: float | None = None,
):
    """
    대형 모델을 BnB 4-bit로 안전하게 로딩.

    Args:
        model_path: 로컬 스냅샷 경로 (HF repo id 아님)
        install_deps: pip install 실행 여부
        clear_module_cache_for: transformers_modules 하위 organization 이름
        n_params_b: 모델 파라미터 수(Billion). 지정하면 메모리 예산 사전 검증.

    Returns:
        (model, tokenizer)
    """
    if install_deps:
        _install_deps_if_needed()

    hf_home = os.environ.get("HF_HOME", "/workspace/.cache_hf")
    if clear_module_cache_for:
        _clear_module_cache(hf_home, clear_module_cache_for)

    # 이 시점에서 torch import (env 설정 후)
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoConfig,
        BitsAndBytesConfig,
    )

    # GPU 정보
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("CUDA GPU 없음")

    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {n_gpus}x {torch.cuda.get_device_properties(0).name} ({gpu_vram_gb:.0f}GB)")

    # 메모리 예산 검증
    if n_params_b:
        _check_memory_budget(n_params_b, n_gpus, gpu_vram_gb)

    # Config + fp8 quantization_config 제거
    print(f"[load] config from {model_path}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(cfg, "quantization_config"):
        print(f"[load] removing config.quantization_config: {cfg.quantization_config}")
        del cfg.quantization_config

    # 수동 device_map 생성
    from pipeline.templates.bnb_manual_device_map import build_device_map, summary
    device_map = build_device_map(cfg, n_gpus=n_gpus)
    print(f"[load] {summary(device_map)}")

    # BnB config (meta tensor 버그 회피)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )

    # Tokenizer
    print(f"[load] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 모델 로딩
    print(f"[load] model (this may take 30+ minutes for 500GB+ models)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    elapsed = time.time() - t0
    total_vram = sum(
        torch.cuda.memory_allocated(i) / 1e9 for i in range(n_gpus)
    )
    n_layers = getattr(cfg, "num_hidden_layers", "?")
    print(
        f"[load] DONE in {elapsed:.0f}s. "
        f"{n_layers} layers, total VRAM {total_vram:.1f}GB"
    )

    return model, tokenizer


def make_safe_hook(callback):
    """
    Multi-GPU 안전 forward hook 래퍼.

    callback: (h_in_cpu: torch.Tensor, h_out_cpu: torch.Tensor) -> None
    """
    def _hook(module, inp, out):
        # 훅 진입 즉시 CPU로 내려서 device mismatch 방지
        h_out = (out[0] if isinstance(out, tuple) else out).detach().cpu().float()
        h_in = inp[0].detach().cpu().float()
        callback(h_in, h_out)

    return _hook
