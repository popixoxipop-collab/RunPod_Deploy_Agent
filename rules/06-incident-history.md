# Incident History

주요 크래시 유형과 대응 패턴. 원인 / 증상 / 해결만 기록.

| # | 유형 | 교훈 |
|---|------|------|
| 1 | BnB 4-bit 대형 모델 forward 첫 호출 meta tensor 크래시 | `max_memory`에 `"cpu"` 엔트리 금지, 수동 device_map 필수 |
| 2 | BnB 4-bit 대형 모델 로딩 중 GPU 단편화 OOM | `expandable_segments=True` 필수 |
| 3 | Spot instance가 장기 작업 중 회수됨 | 장기 작업 on-demand 강제 |
| 4 | accelerate가 원본 dtype 기준 device_map 오산 → CPU offload 자동 트리거 | 수동 device_map |
| 5 | multi-GPU forward hook device mismatch 위험 | `.detach().cpu().float()` 패턴 |
| 6 | fp8 체크포인트 BnB 4-bit 비호환 | BF16 변환본 + `torch_dtype=float16` |
| 7 | RunPod NFS I/O error (os error 5) | `HF_HUB_DISABLE_XET=1` |
| 8 | `hf_transfer` 대용량 다운로드 futex deadlock | `hf_transfer` OFF + `max_workers=8` |
| 9 | BnB 4-bit 원본 BF16 저장 위한 볼륨 부족 | 볼륨 ≥ params_B × 2 × 1.1 GB |
| 10 | Network volume 자동화 cleanup 중 실수 삭제 | `deleteNetworkVolume` 차단 hook |
| 11 | GPTQ + auto-gptq Triton 무한 컴파일 | `DISABLE_EXLLAMA=1` + 버전 고정 |
| 12 | `transformers>=4.51` 이 `set_submodule` 요구 | PyTorch 2.5+ 업그레이드 또는 `transformers<4.50` 핀 |
| 13 | `bitsandbytes>=0.49` 가 `libnvJitLink.so.13` 요구 | `LD_LIBRARY_PATH` 설정 또는 버전 downgrade |
| 14 | HF cache 중복 (`hub/` 하위 복사본 생성) | `HF_HOME` + `HUGGINGFACE_HUB_CACHE` 동일 경로 |
| 15 | Idle pod 장시간 방치 과금 | Idle monitor 자동화 |
| 16 | `rsync --exclude=data/` 가 모든 레벨 제외 | `--exclude=/data/` (슬래시 prefix) |
| 17 | vCPU 초과 병렬 프로세스 → SSH 불능 | `MAX_PARALLEL = vCPU_count` |
| 18 | 기존 pod 확인 없이 중복 pod 생성 | Pre-deploy triage |
| 19 | RunPod API `api-key:` 헤더 → `myself: null` | `Authorization: Bearer` |
| 20 | 대형 이미지 pull 타임아웃 | SECURE 우선, 경량 이미지 fallback |
| 21 | `podFindAndDeployOnDemand` `gpuCount` 누락 → SUPPLY_CONSTRAINT | `gpuCount` 필수 |
| 22 | SSH `wait_for_pod` 기본값 짧아 이미지 pull 중 타임아웃 | 기본값 1200초 |
| 23 | `gpuDisplayName` 필드 삭제됨 | 필드 제거 |
| 24 | `apt-get install rsync` silent fail | returncode + `which rsync` 검증 |
| 25 | `nohup` SSH timeout → 중복 프로세스 | `Popen(start_new_session=True)` |
| 26 | 공유 인프라 load avg 급증 | rescue Pod 즉시 대체 |
| 27 | Pod terminate 후 로그 유실 | `fetch_logs → terminate` 순서 |
| 28 | 과도한 병렬 프로세스 → CPU 100% | `MAX_PARALLEL=vCPU` |
| 29 | 과도한 pod 생성으로 GPU util 낭비 | 모델 크기 기반 pod 수 계산 |
| 30 | 완료된 job 전체 재배포 | 로컬 완료 확인 후 누락분만 배포 |
| 31 | launcher crash → 실행중 Pod terminate | `try/except` 분리 |
| 32 | 세션 전환 시 모니터 agent가 기존 Pod 무시 | 세션 시작 시 기존 pod 상태 확인 |

---

## 상세 기록 (주요 크래시 패턴)

### BnB meta tensor 크래시

**시퀀스**:
```
1. 스크립트 시작, `expandable_segments=True` 적용
2. pip install (`transformers`, `bitsandbytes` 등 버전 핀)
3. 샤드 로딩 완료
4. 첫 `model(**inputs)` 호출
5. CRASH: NotImplementedError: Cannot copy out of meta tensor; no data!
```

**스택 트레이스 핵심**:
```
File "bitsandbytes/nn/modules.py", in to
    self.quant_state.to(device)
File "bitsandbytes/functional.py", in to
    self.code = self.code.to(device)
```

**원인**: `max_memory = {0..N: "XGiB", "cpu": "YGiB"}` 에 `"cpu"` 엔트리 존재 →
accelerate가 원본 dtype 크기 기준 계산 → 일부 레이어 `"cpu"`로 device_map 할당 →
로딩은 성공 → 첫 model call 시 accelerate가 `offload=True` pre_forward로
`set_module_tensor_to_device` 호출 → BnB meta tensor 버그.

**해결 시도 1 (실패)**: `"cpu"` 엔트리만 제거.
→ transformers validator가 "Some modules are dispatched on the CPU or the disk" 로 거부.

**해결 시도 2 (성공)**: 수동 device_map으로 모든 레이어를 정수 GPU ID에 명시 배치.
```python
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
```

---

### GPU 단편화 OOM

**증상**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB.
GPU N has a total capacity of X GiB of which Y MiB is free.
This process has X GiB memory in use.
Of the allocated memory N GiB is allocated by PyTorch,
and M GiB is reserved by PyTorch but unallocated.
If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

**원인**: PyTorch 기본 segment allocator가 BnB의 BF16 shard load / 4-bit quantize /
BF16 해제 사이클에서 파편화시킴. `max_memory` 한도 근처에 도달하면 workspace 여유가
얼마 남지 않는데, 파편화로 연속 블록 못 찾음.

**해결**: `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` 를
스크립트 최상단(torch import 전)에 추가. PyTorch 2.1+가 가변 크기 segment 지원.

---

### Spot instance 회수

**시퀀스**:
```
- spot pod 생성
- SSH 접속 완료, 패치 준비 중
- pod terminate 감지 (운영사 회수, 고지 없음)
```

**원인**: `interruptible: true` 선택 시 RunPod 운영사가 언제든 회수 가능.
현물 가격 변동 + 할당 최적화 등 이유로 수분~수시간 사이 종료.

**기대값 비교 (예시)**:
- On-demand: `X × T`
- Spot: `Y × T × (1 + p × 1.5)` (p=중단 확률)
- p > 20% 면 on-demand가 더 유리

**결론**: 1h+ 작업은 무조건 on-demand.

---

### Network volume 삭제 사고

**시퀀스**:
```
- 대형 모델이 다운로드된 볼륨 존재
- DC 변경 필요한 작업
- 자동 cleanup 단계에서 볼륨 삭제
- 모델 데이터 소멸, 재다운로드 필요
```

**예방**: Hook에서 `deleteNetworkVolume` 명령 차단. 명시적 승인 있을 때만 허용.
```python
if 'deleteNetworkVolume' in command:
    block("Network Volume 삭제 금지 - 명시적 승인 필요")
```
