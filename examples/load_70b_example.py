"""
70B급 모델 BnB 4-bit 로딩 예제 (generic).

실행 전:
    export RUNPOD_API_KEY=$(cat ~/.runpod_api_key)
    python pipeline/preflight-guard.py examples/load_70b_example.py
"""
# 최상단 필수 환경 변수 — torch import 전
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/workspace/.cache_hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache_hf"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache_hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
import subprocess
import time

# Pod 재시작 후 pip env 리셋되므로 매번 재설치
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.51.3",
     "accelerate==1.13.0",
     "bitsandbytes==0.49.2",
     "scikit-learn"],
    check=True,
)

import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)

# bitsandbytes SCB 몽키패치 (Colab Python 3.12 + device_map SCB 버그 예방)
if not getattr(bnb.nn.Linear8bitLt, "_scb_patched", False):
    _orig_save = bnb.nn.Linear8bitLt._save_to_state_dict
    def _safe_save(self, destination, prefix, keep_vars):
        if not hasattr(self, "SCB") or self.SCB is None:
            self.SCB = torch.zeros(self.weight.shape[0], dtype=torch.int8)
        return _orig_save(self, destination, prefix, keep_vars)
    bnb.nn.Linear8bitLt._save_to_state_dict = _safe_save
    bnb.nn.Linear8bitLt._scb_patched = True

# 모델 경로 (로컬 스냅샷)
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
LOCAL_PATH = "/workspace/.cache_hf/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/XXX"

# ── GPU 확인 ────────────────────────────────────────
n_gpus = torch.cuda.device_count()
print(f"\nGPU count: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} ({p.total_memory/1e9:.0f}GB)")

# ── Config 로딩 + fp8 quantization_config 제거 ──────
_cfg = AutoConfig.from_pretrained(LOCAL_PATH, trust_remote_code=True)
if hasattr(_cfg, "quantization_config"):
    print(f"제거: {_cfg.quantization_config}")
    del _cfg.quantization_config

# ── 수동 device_map 생성 ───────────────────────────
# 70B 모델 → 4-bit ~40GB → A100 40GB x2 또는 80GB x1로 충분하지만
# 여기선 multi-GPU 예제를 보여주기 위해 분배
n_layers = _cfg.num_hidden_layers  # Llama 3.1 70B = 80 layers
layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
layers_per_gpu.reverse()

custom_device_map = {"model.embed_tokens": 0}
idx = 0
for gpu_id, n in enumerate(layers_per_gpu):
    for _ in range(n):
        custom_device_map[f"model.layers.{idx}"] = gpu_id
        idx += 1
custom_device_map["model.norm"] = n_gpus - 1
custom_device_map["lm_head"] = n_gpus - 1

print(f"device_map: {n_layers} layers x {n_gpus} GPUs -> {layers_per_gpu}")

# ── BnB 4-bit config (meta tensor 버그 회피) ────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

# ── 로딩 ────────────────────────────────────────────
print(f"\n로딩: {MODEL_ID}")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(LOCAL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH,
    config=_cfg,
    quantization_config=bnb_config,
    device_map=custom_device_map,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.eval()

elapsed = time.time() - t0
total_vram = sum(torch.cuda.memory_allocated(i)/1e9 for i in range(n_gpus))
print(f"로딩 완료: {elapsed:.0f}s, {n_layers} layers, VRAM {total_vram:.1f}GB")

# ── 테스트 inference ───────────────────────────────
print("\n테스트 inference:")
first_device = model.model.embed_tokens.weight.device
inputs = tok("Hello, world!", return_tensors="pt").to(first_device)
with torch.no_grad():
    out = model(**inputs)
print(f"logits shape: {out.logits.shape}")
print("OK")
