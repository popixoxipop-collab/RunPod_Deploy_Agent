"""
환경 변수 템플릿 — 대형 모델 로딩 전 반드시 설정.

사용법: 스크립트 최상단에 `from pipeline.templates.env_setup import setup_env`
       그 후 `setup_env()` 호출. torch import **전에**.
"""
import os


def setup_env(cache_dir: str = "/workspace/.cache_hf") -> None:
    """대형 모델 안전 로딩 환경 변수 일괄 설정."""

    # 1. 메모리 단편화 방지 (대형 BnB 로딩 필수)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 2. HF cache 통일 (hub/ 복사본 방지)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    # 3. NFS I/O 에러 방지 (RunPod 볼륨)
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    # 4. Tokenizer 병렬 경고 끄기
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 5. bitsandbytes CUDA lib 경로 (0.49+ + CUDA 12.4 조합 시 필요)
    cu13_lib = "/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib"
    if os.path.isdir(cu13_lib):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if cu13_lib not in existing:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{cu13_lib}:{existing}" if existing else cu13_lib
            )


if __name__ == "__main__":
    setup_env()
    print("[env_setup] applied:")
    for k in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_DISABLE_XET",
        "LD_LIBRARY_PATH",
    ):
        print(f"  {k}={os.environ.get(k, '(unset)')}")
