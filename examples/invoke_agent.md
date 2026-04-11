# 에이전트 호출 예제

Claude Code에서 RunPod Deploy Agent를 호출하는 실전 예제.

---

## 예제 1: 70B 모델 로딩

```
/agent runpod-deploy

Llama 3.1 70B Instruct를 RunPod A100 80GB 2장에서 BnB 4-bit로 로딩하는
스크립트 작성해줘.
```

**에이전트가 할 일**:
1. 메모리 예산 계산 (70B × 0.5 × 1.15 = 40.25 GB < 2 × 80 × 0.85 = 136 GB ✓)
2. 스크립트 작성 (templates 사용, 수동 device_map 포함)
3. `runpod-preflight` 로 정적 분석
4. 사용자 승인 후 pod 생성 (on-demand, volume 연결)
5. 스크립트 업로드 + 백그라운드 실행
6. 60초 후 double-check
7. 결과 rsync → pod stop

---

## 예제 2: 기존 pod 재활용

```
/agent runpod-deploy

이전에 돌리던 pod 상태 확인하고, idle면 같은 볼륨에 새 스크립트 올려줘.
```

**에이전트가 할 일**:
1. `runpod-deploy --list` 로 기존 pod 조회
2. SSH 접속 → `ps aux` 로 실행 상태 확인
3. idle 판정 시 새 스크립트 rsync → 실행
4. 재생성 없이 재활용

---

## 예제 3: 크래시 진단

```
/agent runpod-deploy

방금 로딩 중에 터졌는데 로그 이것 좀 봐줘: [로그 붙여넣기]
```

**에이전트가 할 일**:
1. 스택 트레이스 분석 → 알려진 크래시 패턴 매칭
2. `rules/01-crash-catalog.md` 에서 해당 카테고리 찾기
3. 수정 방법 제시 + 파이프라인 검증
4. 로그는 terminate 전에 먼저 rsync

---

## 예제 4: 대형 모델 전체 워크플로

```
/agent runpod-deploy

대형 모델을 BnB 4-bit로 A100 80GB 여러 장에 로딩하는 스크립트를
작성하고 실행해줘. 결과는 로컬에 저장.
```

**에이전트가 할 일 (전체 시퀀스)**:

### Phase 0: Plan
- 메모리: `params_b × 0.5 × 1.15` GB (4-bit + overhead)
- 예산: `n_gpus × vram_gb × 0.85` GB (15% workspace 여유)
- 볼륨: `params_b × 2 × 1.1` GB (BnB는 원본 BF16 필요)
- 시간/비용 추정
- **사용자 승인 대기**

### Phase 1: 스크립트 작성 (templates 사용)

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/workspace/.cache_hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache_hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch, time, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.48.3", "accelerate==1.13.0",
                "bitsandbytes==0.49.2"], check=True)

from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

LOCAL_PATH = "/workspace/.cache_hf/models--ORG--MODEL/snapshots/XXX"

# Config + 불필요한 quantization_config 제거
cfg = AutoConfig.from_pretrained(LOCAL_PATH, trust_remote_code=True)
if hasattr(cfg, "quantization_config"):
    del cfg.quantization_config

# 수동 device_map (R12)
n_layers = cfg.num_hidden_layers
n_gpus = torch.cuda.device_count()
layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
layers_per_gpu.reverse()

device_map = {"model.embed_tokens": 0}
idx = 0
for gpu_id, n in enumerate(layers_per_gpu):
    for _ in range(n):
        device_map[f"model.layers.{idx}"] = gpu_id
        idx += 1
device_map["model.norm"] = n_gpus - 1
device_map["lm_head"] = n_gpus - 1

# BnB (meta tensor 버그 회피)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

# 로딩
t0 = time.time()
tok = AutoTokenizer.from_pretrained(LOCAL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH, config=cfg,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.eval()
print(f"Loaded in {time.time()-t0:.0f}s")

# 이후 사용자 작업 로직
...
```

### Phase 2: 정적 분석
```bash
runpod-preflight /workspace/script.py
# [OK] script.py
```

### Phase 3: 기존 pod 확인
```bash
runpod-deploy --list
# (pod 없음)
```

### Phase 4: Pod 생성
```bash
runpod-deploy \
    --gpu-type "NVIDIA A100-SXM4-80GB" \
    --gpu-count N \
    --volume-id vol_xxxx \
    --datacenter US-MD-1 \
    --public-key "$(cat ~/.ssh/id_ed25519.pub)" \
    --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# [OK] Pod 생성됨: <pod_id>
```

### Phase 5: 스크립트 업로드 + 실행
```bash
RUNPOD_AGENT=1 scp -P <port> /workspace/script.py root@<ip>:/workspace/
RUNPOD_AGENT=1 ssh -p <port> root@<ip> \
    "nohup python /workspace/script.py > /workspace/run.log 2>&1 </dev/null &"
```

### Phase 6: 60초 double-check (R9)
```bash
sleep 60
RUNPOD_AGENT=1 ssh -p <port> root@<ip> "ps aux | grep python | grep -v grep"
```

### Phase 7: 모니터링
```bash
# Idle monitor 백그라운드 시작
runpod-idle-monitor --action stop --poll-interval-sec 300 &
```

### Phase 8: 완료 후 cleanup
```bash
# 결과 rsync
RUNPOD_AGENT=1 scp -P <port> root@<ip>:/workspace/results/* ./results/
# Pod 종료
runpod-deploy --terminate <pod_id>
# Volume 유지 여부 사용자 확인
```

---

## 금지 예제 (에이전트가 거절해야 함)

### 사용자가 spot 요청
```
/agent runpod-deploy
Spot 인스턴스로 3시간짜리 로딩 돌려줘
```

**에이전트 응답**:
> 1시간+ 작업에 spot 사용 금지 (R13). 뺏기면 시간 전체 손실.
> On-demand로 진행할까요? 추가 비용 약 $X.

### 사용자가 volume 삭제 요청 없이 삭제 자동화
```
/agent runpod-deploy
작업 끝나면 pod + 볼륨 전부 정리해줘
```

**에이전트 응답**:
> Pod은 terminate 가능하지만, volume 삭제는 사용자 명시적 "삭제" 확인 필요 (R14).
> 볼륨 유지 시 시간당 저장 비용만 과금됩니다. 삭제할까요?
