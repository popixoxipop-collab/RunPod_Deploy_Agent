# BnB 4-bit 대형 모델 안전 레시피

70B~671B 모델을 bitsandbytes 4-bit로 로딩할 때 반드시 따라야 할 패턴.

---

## 검증된 환경

| 항목 | 버전 |
|------|------|
| PyTorch | 2.4.1+cu124 (또는 2.6.0+cu124) |
| transformers | 4.48.3 (R1/V3 등 custom modeling_*.py 모델) 또는 4.51+ (표준 HF 모델) |
| bitsandbytes | 0.49.2 (주의: 아래 "BnB 0.49.2 버그 주의" 참고) |
| accelerate | 1.13.0 |
| CUDA | 12.4 |
| 이미지 | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

---

## 필수 환경변수

```python
import os
# 최상단, torch import 전
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/workspace/.cache_hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache_hf"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache_hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # NFS I/O 에러 방지
```

---

## 필수 패키지 핀

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.48.3",  # 또는 특정 모델 요구 버전
    "accelerate==1.13.0",
    "bitsandbytes==0.49.2",
    "scikit-learn",
], check=True)
```

> **주의**: `bitsandbytes>=0.44.0`처럼 느슨한 제약 사용 금지. 재배포마다 다른 버전 설치되어 비결정적 버그 유발.

---

## BnB Config (메타 텐서 버그 회피)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # fp8 입력 모델엔 fp16 강제
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,  # double_quant는 meta tensor 이슈 유발 가능
    # NOTE: llm_int8_enable_fp32_cpu_offload=True 넣지 말 것
    #       (CPU offload path가 BnB 0.49.2 meta tensor bug 트리거)
)
```

---

## 수동 device_map 생성 (핵심)

`device_map="auto"` 금지. accelerate가 BF16 원본 크기 기준으로 오산하여
CPU 오프로드를 자동 트리거 → meta tensor 버그.

```python
from transformers import AutoConfig

# 1. Config 로딩 + fp8 quantization_config 제거 (필요 시)
_cfg = AutoConfig.from_pretrained(LOCAL_PATH, trust_remote_code=True)
if hasattr(_cfg, 'quantization_config'):
    print(f"제거: {_cfg.quantization_config}")
    del _cfg.quantization_config

# 2. 레이어 수 + GPU 수 기반 분배
n_layers = _cfg.num_hidden_layers
n_gpus = torch.cuda.device_count()
layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
layers_per_gpu.reverse()  # 앞쪽 GPU(embed 포함)에 더 많이

# 3. device_map dict 구성 (모든 값은 정수 GPU ID, "cpu"/"disk" 금지)
custom_device_map = {"model.embed_tokens": 0}
idx = 0
for gpu_id, n in enumerate(layers_per_gpu):
    for _ in range(n):
        custom_device_map[f"model.layers.{idx}"] = gpu_id
        idx += 1
custom_device_map["model.norm"] = n_gpus - 1
custom_device_map["lm_head"] = n_gpus - 1

print(f"device_map: {n_layers} layers across {n_gpus} GPUs → {layers_per_gpu}")
```

> **주의**: 위는 LLaMA/Qwen/DeepSeek 스타일 (`model.layers.{i}`) 기준.
> 다른 아키텍처는 모듈 이름이 다를 수 있음. 빈 모델 inspect로 확인:
> ```python
> from accelerate import init_empty_weights
> with init_empty_weights():
>     m = AutoModelForCausalLM.from_config(_cfg, trust_remote_code=True)
> for n, _ in m.named_children():
>     print(n)
> ```

---

## 모델 로딩

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH,                     # 로컬 스냅샷 경로 권장 (hub/ 복사본 방지)
    config=_cfg,
    quantization_config=bnb_config,
    device_map=custom_device_map,   # 수동 매핑
    trust_remote_code=True,
    torch_dtype=torch.float16,      # fp8 원본이면 강제 필수
)
model.eval()
```

---

## Hook 함수 (multi-GPU 안전)

```python
def _hook(module, inp, out):
    # 훅 진입 즉시 CPU로 내려서 device mismatch 방지
    h_out = (out[0] if isinstance(out, tuple) else out).detach().cpu().float()
    h_in = inp[0].detach().cpu().float()
    delta = h_out - h_in
    # 이후 CPU 연산만
    ...
```

---

## 메모리 예산 검증 (로딩 전)

```python
def check_memory_budget(n_params_b, n_gpus, gpu_vram_gb):
    """BnB 4-bit 메모리 예산 검증."""
    model_size_gb = n_params_b * 0.5 * 1.15  # 4-bit + 15% 메타
    gpu_budget_gb = n_gpus * gpu_vram_gb * 0.85  # 15% workspace 여유
    
    if model_size_gb > gpu_budget_gb:
        raise RuntimeError(
            f"VRAM 부족: 모델 {model_size_gb:.0f}GB > 예산 {gpu_budget_gb:.0f}GB. "
            f"필요 GPU: {int(n_params_b * 0.5 * 1.15 / (gpu_vram_gb * 0.85)) + 1}"
        )
    
    print(f"[OK] 모델 {model_size_gb:.0f}GB, 예산 {gpu_budget_gb:.0f}GB, "
          f"여유 {gpu_budget_gb - model_size_gb:.0f}GB")

check_memory_budget(n_params_b=671, n_gpus=7, gpu_vram_gb=80)
```

---

## 볼륨 크기 계산

```python
def volume_size_gb_for_bnb_4bit(n_params_b, source_dtype="bf16"):
    """
    BnB 4-bit은 원본 bf16/fp16 먼저 다운로드 필요.
    fp8 원본은 × 1, bf16/fp16은 × 2 바이트.
    """
    bytes_per_param = 2 if source_dtype in ("bf16", "fp16") else 1
    return int(n_params_b * bytes_per_param * 1.1)  # 10% 여유

vol_gb = volume_size_gb_for_bnb_4bit(671, "bf16")
print(f"필요 볼륨 크기: {vol_gb} GB")  # 1476 GB → 1500 GB 생성
```

---

## 다운로드 안전 패턴

```python
from huggingface_hub import snapshot_download
import time

MAX_RETRIES = 20
for attempt in range(MAX_RETRIES):
    try:
        snapshot_download(
            repo_id=REPO_ID,
            cache_dir=os.environ["HF_HOME"],
            max_workers=8,          # hf_transfer 없이 기본 다운로더
            local_dir=None,         # hub/ 구조 유지
            resume_download=True,   # 중단 시 이어받기
        )
        print("다운로드 완료")
        break
    except Exception as e:
        print(f"attempt {attempt+1}/{MAX_RETRIES} 실패: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(30)
```

---

## BnB 0.49.2 버그 주의사항

### 1. meta tensor in `quant_state.code`

**증상**: 로딩 후 forward 첫 호출에서 `NotImplementedError: Cannot copy out of meta tensor`

**트리거**: CPU offload 경로 (accelerate `AlignDevicesHook.pre_forward` + `offload=True`)

**회피**: 위 "수동 device_map" 패턴 엄격 준수 + `bnb_4bit_use_double_quant=False`

### 2. `libnvJitLink.so.13` 요구

**증상**: import 시 `OSError: libnvJitLink.so.13: cannot open shared object file`

**트리거**: bitsandbytes 0.49+ + CUDA < 12.9

**회피**:
```bash
pip install nvidia-nvjitlink-cu13
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```
또는 `bitsandbytes<0.49` 핀.

### 3. BF16 원본의 on-the-fly 양자화 OOM

**증상**: 로딩 중 단편화 OOM (21+ GB "reserved but unallocated")

**트리거**: 기본 PyTorch segment allocator + BnB 반복 alloc/free

**회피**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (필수)

---

## 전체 레시피 (바로 복붙 가능)

`pipeline/templates/large_model_loader.py` 참고.
