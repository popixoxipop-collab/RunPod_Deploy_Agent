# Incident History

실제 발생 사건의 상세 기록. `CHANGELOG.md`보다 자세함.

| # | 날짜 | 모델 규모 | 사건 | 손실 | 교훈 |
|---|------|----------|------|------|------|
| 1 | 2026-04-11 | 671B | BnB 0.49.2 meta tensor 버그 (forward 첫 호출) | ~$33 | max_memory에 "cpu" 엔트리 금지, 수동 device_map 필수 |
| 2 | 2026-04-10 | 671B | GPU 단편화 OOM (샤드 120/163) | ~$18 | expandable_segments=True 필수 |
| 3 | 2026-04-10 | 671B | Spot instance 뺏김 (8x A100 80GB) | ~$3 + 시간 손실 | 1h+ 작업 on-demand 강제 |
| 4 | 2026-04-10 | 671B | accelerate BF16 기준 device_map 오산 → CPU offload 트리거 | 시간 손실 | 수동 device_map |
| 5 | 2026-04-10 | - | multi-GPU hook device mismatch 위험 | 예방 | .detach().cpu().float() 패턴 |
| 6 | 2026-04-10 | 671B | fp8 체크포인트 BnB 4-bit 비호환 | 시간 손실 | BF16 변환본 + torch_dtype=float16 |
| 7 | 2026-04-10 | 235B+ | RunPod NFS I/O error (os error 5) | 시간 손실 | HF_HUB_DISABLE_XET=1 |
| 8 | 2026-04-10 | 671B | hf_transfer futex deadlock | 시간 손실 | hf_transfer OFF + max_workers=8 |
| 9 | 2026-04-09 | 235B | 볼륨 크기 부족 (BnB 4-bit 원본 BF16) | 시간 손실 | volume >= params_B x 2 x 1.1 |
| 10 | 2026-04-09 | - | Network volume 실수 삭제 | 모델 재다운 | deleteNetworkVolume 차단 hook |
| 11 | 2026-04-09 | 235B | Qwen3 GPTQ auto-gptq Triton 무한 컴파일 | 시간 손실 | DISABLE_EXLLAMA=1 + 버전 고정 |
| 12 | 2026-04-09 | - | transformers 4.51+ set_submodule 요구 | 시간 손실 | PyTorch 2.5+ 업그레이드 |
| 13 | 2026-04-09 | - | bitsandbytes 0.49+ libnvJitLink.so.13 not found | 시간 손실 | LD_LIBRARY_PATH 또는 버전 downgrade |
| 14 | 2026-04-07 | - | HF cache 중복 (hub/ 하위 복사본) | 볼륨 초과 | HF_HOME + HUGGINGFACE_HUB_CACHE 동일 |
| 15 | 2026-04-06 | - | Idle pod 3시간 방치 | ~$4 | Idle monitor 자동화 |
| 16 | 2026-03-23 | - | rsync --exclude=data/ 오해 (모든 레벨 제외) | 데이터 누락 | --exclude=/data/ (슬래시 prefix) |
| 17 | 2026-03-21 | - | vCPU 초과 병렬 프로세스 (SSH 불능) | 시간 손실 | MAX_PARALLEL = vCPU_count |
| 18 | 2026-03-21 | - | 중복 Pod 생성 (기존 확인 안 함) | $1.76/hr 중복 | Pre-deploy triage |
| 19 | 2026-03-21 | - | RunPod API `api-key:` 헤더 → myself: null | 시간 손실 | Authorization: Bearer |
| 20 | 2026-03-20 | - | 이미지 pull 60분+ 타임아웃 | 시간 손실 | SECURE 우선, 경량 이미지 fallback |
| 21 | 2026-03-13 | - | `podFindAndDeployOnDemand` gpuCount 누락 → SUPPLY_CONSTRAINT | 배포 실패 | gpuCount 필수 |
| 22 | 2026-03-13 | - | SSH wait_for_pod 300초 초과 | 배포 실패 | 1200초로 상향 |
| 23 | - | - | `gpuDisplayName` 필드 삭제됨 | 배포 실패 | 필드 제거 |
| 24 | - | - | apt-get install rsync silent fail | 배포 실패 | returncode + `which rsync` 검증 |
| 25 | - | - | nohup SSH timeout 35s → 중복 프로세스 | 로그 corruption | `Popen(start_new_session=True)` |
| 26 | - | - | Pod 2 load avg 68→87 (공유 인프라 과부하) | 성능 저하 | rescue Pod 즉시 대체 |
| 27 | - | - | Pod terminate 후 로그 유실 | 실험 결과 손실 | fetch_logs -> terminate 순서 |
| 28 | - | - | 1 Pod x 45 procs → CPU 100% | SSH 불능 | MAX_PARALLEL=vCPU |
| 29 | - | - | 5 Pod x GPU 14% 낭비 | 비용 3배 | 모델 크기 기반 pod 수 계산 |
| 30 | - | - | 완료 9/12 jobs 전체 재배포 | 비용 낭비 | R7 로컬 완료 확인 |
| 31 | - | - | launcher crash → 실행중 Pod terminate | 실험 중단 | try/except 분리 |
| 32 | - | - | 세션 전환 시 모니터 agent가 기존 Pod 무시 | 로그 유실 | R8 |

---

## 상세 기록 (주요 사건)

### #1 — 2026-04-11: BnB meta tensor 버그

**시퀀스**:
```
00:00  스크립트 시작, expandable_segments=True 적용 (최근 크래시 교훈)
00:05  pip install transformers==4.48.3, bitsandbytes 0.49.2
03:42  163/163 샤드 로딩 성공! (이전 OOM 지점 120/163 통과)
       "Loaded in 13769s. 61 layers, hidden=7168"
03:42  첫 `model(**inputs)` 호출
03:42  CRASH: NotImplementedError: Cannot copy out of meta tensor; no data!
```

**스택 트레이스 핵심**:
```
File "bitsandbytes/nn/modules.py", line 348, in to
    self.quant_state.to(device)
File "bitsandbytes/functional.py", line 595, in to
    self.code = self.code.to(device)
```

**원인**: `max_memory = {0..5: "65GiB", "cpu": "200GiB"}` 에 `"cpu"` 엔트리 존재 →
accelerate가 BF16 1342 GB 기준 계산 → 일부 레이어 `"cpu"`로 device_map 할당 →
로딩은 성공 (BnB가 실제로는 GPU에 quantize) → forward 첫 호출 시 accelerate가
`offload=True` pre_forward로 `set_module_tensor_to_device` 호출 → BnB meta tensor 버그.

**해결 시도 1 (실패)**: `"cpu"` 엔트리만 제거, `max_memory = {0..6: "72GiB"}` 유지.
→ transformers validator가 "Some modules are dispatched on the CPU or the disk" 로 거부.

**해결 시도 2 (성공)**: 수동 device_map으로 모든 레이어를 정수 GPU ID에 명시 배치.
```python
n_layers = cfg.num_hidden_layers  # 61
n_gpus = 7
layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
layers_per_gpu.reverse()  # [9,9,9,9,9,8,8]

device_map = {"model.embed_tokens": 0}
idx = 0
for gpu_id, n in enumerate(layers_per_gpu):
    for _ in range(n):
        device_map[f"model.layers.{idx}"] = gpu_id
        idx += 1
device_map["model.norm"] = n_gpus - 1
device_map["lm_head"] = n_gpus - 1
```

**손실**:
- 실제 크래시 비용: ~$33 (3시간 42분 × $8.94/hr)
- 전체 세션: ~$80+ (누적 크래시)

---

### #2 — 2026-04-10: GPU 단편화 OOM

**시퀀스**:
```
00:00  스크립트 시작, max_memory={0..4: 76, 5: 72}
00:30  샤드 로딩 10/163
02:20  샤드 120/163 (74%)
02:21  CRASH: GPU 5 OOM, 21.90 GiB reserved but unallocated
```

**로그**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 5 has a total capacity of 79.25 GiB of which 6.94 MiB is free.
Including non-PyTorch memory, this process has 79.23 GiB memory in use.
Of the allocated memory 56.85 GiB is allocated by PyTorch,
and 21.90 GiB is reserved by PyTorch but unallocated.
If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

**원인**: PyTorch 기본 segment allocator가 BnB의 BF16 shard load / 4-bit quantize /
BF16 해제 사이클에서 파편화시킴. GPU5가 max_memory 72 GiB에 도달하면 workspace 8 GB밖에
남지 않는데, 파편화로 연속 20 MB도 못 찾음.

**해결**: `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` 를
스크립트 최상단에 추가. PyTorch 2.1+가 가변 크기 segment 지원.

**손실**: ~$18.

---

### #3 — 2026-04-10: Spot instance 뺏김

**시퀀스**:
```
20:09  8x A100 80GB spot pod 생성 ($7.60/hr)
20:15  SSH 접속 완료, 패치 준비 중
20:20  pod terminate 감지 (이유: 운영사 회수, 고지 없음)
```

**원인**: `interruptible: true` 선택 시 RunPod 운영사가 언제든 회수 가능.
현물 가격 변동 + 할당 최적화 등 이유로 수분~수시간 사이 종료.

**기대값 계산**:
- On-demand $8.94/hr × 3h = $26.82
- Spot $7.60/hr × 3h = $22.80 (중단 없을 때)
- Spot 중단 확률 15% × 3h = $11.40 손실 기대
- 기대값: $22.80 + $11.40 = $34.20 > $26.82

**결론**: 1h+ 작업은 무조건 on-demand.

---

### #10 — 2026-04-09: Network volume 삭제 사고

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
