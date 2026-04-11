# Pre-Deploy Checklist

배포 시작 전 반드시 통과해야 하는 항목. 하나라도 실패하면 배포 금지.

---

## Phase 0: 환경 검증

- [ ] RunPod API 키 준비 (`~/.runpod_api_key` 또는 env)
- [ ] SSH 공개키 확인 (`~/.ssh/id_ed25519.pub`)
- [ ] HuggingFace 토큰 준비 (private 모델 사용 시)
- [ ] 충분한 RunPod 잔액 확인 (예상 비용 × 2 이상)
  - 공식: `hours_estimated × hourly_rate × 2 (safety)`
  - 70B 모델: ~3시간 × $8.94 = ~$27 × 2 = **$54 최소**

---

## Phase 1: 기존 Pod/Volume 확인 (R10)

- [ ] 기존 pod 조회 (중복 생성 방지)
  ```bash
  python pipeline/pod_deploy.py --list
  ```
- [ ] Running pod 발견 시 SSH 체크 → 재활용 또는 terminate
- [ ] 기존 Network volume 확인
- [ ] 볼륨 크기 vs 모델 크기 계산
  - BnB 4-bit: `params_B × 2 × 1.1 GB` 이상 필요
  - GPTQ/pre-quantized: `params_B × 0.5 × 1.1 GB` 이상 필요

---

## Phase 2: 스크립트 정적 분석 (CRITICAL)

```bash
python pipeline/preflight_check.py your_script.py
```

자동 검증 항목 (자세한 내용은 `pipeline/preflight_check.py`):

- [ ] Python syntax OK
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정됨 (대형 모델 로딩)
- [ ] `max_memory`에 `"cpu"` 엔트리 없음 (BnB 4-bit 사용 시)
- [ ] `device_map="auto"` 대신 수동 device_map (70B+ 모델)
- [ ] `BitsAndBytesConfig(bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False)`
- [ ] Hook 함수에서 multi-GPU 텐서 직접 연산 없음 (`.detach().cpu().float()` 사용)
- [ ] `HF_HOME`, `HUGGINGFACE_HUB_CACHE` 동일 경로 설정
- [ ] transformers 버전이 모델 요구 버전 이상
- [ ] `total_mem` → `total_memory` 오타 없음
- [ ] RunPod 스크립트에 `google.colab` import 없음
- [ ] RunPod 스크립트에 `HF_HOME = /workspace/.cache_hf` 설정

---

## Phase 3: Pod 구성

- [ ] GPU 수 계산: 모델 크기 / GPU VRAM × 1.2 (20% 여유)
  - 예: 370 GB (4-bit R1) / 80 GB × 1.2 = 5.55 → 6 GPU
- [ ] GPU 타입 선택
  - 70B+: A100 80GB 또는 H100
  - 30B~70B: A100 40GB 또는 A40 또는 RTX A6000
  - <30B: RTX 3090/4090
- [ ] **`interruptible: false` 명시** (장기 작업 1시간+)
- [ ] Network volume 연결
- [ ] `PUBLIC_KEY` env 포함 (SSH 접속용)
- [ ] 이미지 선택
  - 권장: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  - 경량 대안: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- [ ] `cloudType`: SECURE 우선, fallback ALL

---

## Phase 4: 모델 로딩 사전 계산

### BnB 4-bit 로딩 메모리 budget

```
model_size_4bit ≈ params_B × 0.5 × 1.15  # 0.5 byte/param + 15% 메타
gpu_budget ≈ gpu_count × gpu_vram × 0.85  # 15% workspace 여유
assert model_size_4bit < gpu_budget, "VRAM 부족 → GPU 수 늘리기"
```

예시 (DeepSeek-R1 671B, 7× A100 80GB):
```
model_size = 671 × 0.5 × 1.15 = 385.8 GB
gpu_budget = 7 × 80 × 0.85 = 476.0 GB
→ 여유 90 GB, 안전
```

### 레이어별 분배 (수동 device_map)

```python
n_layers = config.num_hidden_layers
n_gpus = torch.cuda.device_count()
layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
# 예: 61 layers / 7 GPUs = [9, 9, 9, 9, 9, 8, 8]
```

MoE 모델은 대부분 레이어 크기 유사하므로 레이어 수 기반 분배 OK.

---

## Phase 5: 모니터링 설정

- [ ] Watchdog 스크립트 준비 (자동 재시작, max 3회)
- [ ] 로그 경로 설정 (`/workspace/experiment.log`)
- [ ] Idle monitor 활성화 (15분 주기)
- [ ] Cleanup hook: `finally`에서 log rsync → terminate 순서

---

## Phase 6: 배포

```bash
python pipeline/pod_deploy.py \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --gpu-type "NVIDIA A100-SXM4-80GB" \
  --gpu-count 7 \
  --volume-id YOUR_VOLUME_ID \
  --no-spot \
  --datacenter US-MD-1
```

---

## Phase 7: 배포 후 검증 (60초 double-check)

- [ ] SSH 접속 OK
- [ ] 스크립트 실행 중 (`ps aux | grep python`)
- [ ] GPU 메모리 증가 중 (로딩 시작됐는지)
- [ ] 로그 파일 생성됐는지

**60초 지나도 Python 프로세스 없음 → 스크립트 실행 실패, 재시작 필요.**

---

## Phase 8: 실험 완료 후 cleanup

- [ ] 결과 파일 로컬로 rsync (terminate 전에)
- [ ] 로컬 파일 검증 (크기, 포맷)
- [ ] Pod terminate (**순서: log fetch → terminate**)
- [ ] Volume 유지 여부 결정 (명시적 승인 없이 삭제 금지)

---

## 금지 사항 (Anti-Checklist)

- ❌ `device_map="auto"` + `max_memory={"cpu": ...}` 조합 (70B+ 모델)
- ❌ `PYTORCH_CUDA_ALLOC_CONF` 미설정 대형 모델 BnB 로딩
- ❌ Spot instance로 1시간+ 작업
- ❌ Hook 함수에서 multi-GPU 텐서 직접 연산
- ❌ `pip install bitsandbytes` 버전 미핀 (`>=0.44.0` 최소)
- ❌ Terminate 전에 로그 rsync 안 함
- ❌ 기존 pod 확인 없이 신규 생성
- ❌ Network volume 자동 삭제
- ❌ RunPod 스크립트에 `/content/` 경로 (Colab 전용)
- ❌ Colab 스크립트에 `/workspace/` 경로 (RunPod 전용)
- ❌ `MAX_PARALLEL > vCPU` (SSH 불능 유발)

---

## 배포 전 보고 포맷

배포 시작 전 다음 정보를 보고:

```
[Deployment Plan]
- Model:         <MODEL_ID> (<X>B params, ~<Y>GB in 4-bit)
- GPU:           <N>x <GPU_TYPE> (<total_vram>GB total)
- Headroom:      <Y>GB model vs <Z>GB budget → <W>% 여유
- Interruptible: false (on-demand)
- Image:         <IMAGE>
- Volume:        <VOL_ID> (<size>GB)
- Datacenter:    <DC>
- Estimated:     loading <X>h + run <Y>h = <Z>h total
- Cost:          ~$<total>
- Preflight:     PASS (X checks)
```

명시적 승인 후 배포 시작.
