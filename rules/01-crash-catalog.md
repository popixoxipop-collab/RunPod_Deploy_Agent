# Crash Catalog

증상별 색인. CHANGELOG.md는 날짜순, 이 파일은 유형별.

---

## A. 메모리 OOM 크래시

### A1. GPU 단편화 OOM (BnB on-the-fly 양자화)

| 항목 | 내용 |
|------|------|
| 증상 | `GPU X has total capacity 79.25 GiB of which 6.94 MiB is free. PyTorch has N GiB reserved but unallocated` |
| 트리거 | BnB 4-bit 반복 alloc/free로 파편화 |
| 발생 시점 | 로딩 중간 (샤드 60% 이상) |
| 해결 | `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` |
| 예방 | 스크립트 최상단(torch import 전)에 설정 강제 |

### A2. 모델 크기 > GPU 예산 → CPU offload 자동 트리거

| 항목 | 내용 |
|------|------|
| 증상 | `Some modules are dispatched on the CPU or the disk` |
| 트리거 | accelerate가 BF16 원본 크기 기준으로 device_map 계산 |
| 발생 시점 | `from_pretrained()` 시작 시 |
| 해결 | 수동 device_map 생성 (아래 A3 참고) |

### A3. BnB meta tensor 버그 (forward 첫 호출)

| 항목 | 내용 |
|------|------|
| 증상 | `NotImplementedError: Cannot copy out of meta tensor; no data!` |
| 트리거 | `max_memory`에 `"cpu"` 엔트리 + BnB 0.49.2 + accelerate offload hook |
| 발생 시점 | 로딩 성공 후 첫 forward pass |
| 원인 | `quant_state.code`가 meta tensor인데 `set_module_tensor_to_device`가 호출됨 |
| 해결 | 수동 device_map (모든 레이어 정수 GPU ID로) + `"cpu"` 엔트리 제거 |
| 추가 | `bnb_4bit_use_double_quant=False` |

---

## B. 라이브러리 호환성

### B1. `libnvJitLink.so.13` not found

| 항목 | 내용 |
|------|------|
| 트리거 | bitsandbytes 0.49+ + CUDA < 12.9 |
| 해결 | `pip install nvidia-nvjitlink-cu13` + `LD_LIBRARY_PATH` 설정 |
| 대안 | `bitsandbytes<0.49` 핀 고정 |

### B2. `set_submodule` 미존재

| 항목 | 내용 |
|------|------|
| 트리거 | `transformers>=4.51.0` + PyTorch < 2.5 |
| 해결 | PyTorch 2.5+ 업그레이드 (torchvision 함께) |
| 대안 | `transformers<4.50` 핀 고정 |

### B3. `torchvision::nms operator not found`

| 항목 | 내용 |
|------|------|
| 트리거 | torch 단독 업그레이드, torchvision 구버전 잔존 |
| 해결 | `pip install torch torchvision --upgrade` 동시 실행 |

### B4. `rshift_cuda` Half 에러 (GPTQ)

| 항목 | 내용 |
|------|------|
| 트리거 | `auto-gptq==0.7.1` + PyTorch 2.4 |
| 해결 | `qlinear_cuda_old.py` line 296(qzeros), 311(qweight)에 `.to(torch.int32)` 패치 |
| 대안 | `GPTQConfig(bits=4, disable_exllama=True)` + `DISABLE_EXLLAMA=1` |

### B5. Triton 커널 무한 컴파일

| 항목 | 내용 |
|------|------|
| 증상 | CPU 2000%+, GPU 0%, I/O 거의 0, 15분+ 멈춤 |
| 트리거 | `auto-gptq` CUDA ext 미설치 → Triton fallback |
| 해결 | CUDA ext 설치 또는 `DISABLE_EXLLAMA=1`로 Triton 우회 |

---

## C. Device / Hook 이슈

### C1. Hook device mismatch

| 항목 | 내용 |
|------|------|
| 증상 | `RuntimeError: Expected all tensors to be on the same device` |
| 트리거 | multi-GPU에서 `register_forward_hook` 내부 GPU 연산, `out`과 `inp[0]`이 다른 GPU |
| 해결 | 훅 진입 즉시 `.detach().cpu().float()` → CPU에서 연산 |

### C2. `output_hidden_states=True` 대형 모델 RAM 폭발

| 항목 | 내용 |
|------|------|
| 증상 | OOM Killed, 또는 시스템 swap 100% |
| 트리거 | 70B+ 모델 × 전 레이어 hidden states 동시 보관 |
| 해결 | `register_forward_hook`으로 레이어별 순차 처리 |

---

## D. 모델 포맷 호환성

### D1. fp8 체크포인트 + BnB 4-bit 비호환

| 항목 | 내용 |
|------|------|
| 증상 | `Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.float8_e4m3fn` |
| 트리거 | DeepSeek-R1/V3 공식 체크포인트 (fp8 block-scaled) |
| 해결 | 커뮤니티 BF16 변환본 사용 + config에서 `quantization_config` 제거 + `torch_dtype=torch.float16` 강제 |

### D2. `quantization_config` 자동 감지 conflict

| 항목 | 내용 |
|------|------|
| 증상 | `Unknown quantization type: fp8` (transformers 4.48.x) |
| 트리거 | config.json에 `quantization_config: fp8` 있는데 우리가 BnB 적용 |
| 해결 | `_cfg = AutoConfig...; del _cfg.quantization_config; from_pretrained(config=_cfg, ...)` |

---

## E. 다운로드 이슈

### E1. NFS I/O error (os error 5)

| 항목 | 내용 |
|------|------|
| 증상 | `IO Error: Input/output error (os error 5)` |
| 트리거 | `snapshot_download` + RunPod NFS + xet 프로토콜 |
| 해결 | `HF_HUB_DISABLE_XET=1` + `max_workers=2` + retry loop 20회 |

### E2. hf_transfer futex deadlock

| 항목 | 내용 |
|------|------|
| 증상 | 초기 770 MB/s → 수 분 후 0 MB/s, process `Dl` state |
| 트리거 | `HF_HUB_ENABLE_HF_TRANSFER=1` + `max_workers=8` |
| 해결 | `hf_transfer` 비활성화 + `max_workers=8` (기본 downloader) |

### E3. 볼륨 크기 부족

| 항목 | 내용 |
|------|------|
| 증상 | `No space left on device` 다운로드 중간 |
| 트리거 | BnB 4-bit은 BF16 원본 필요 (디스크) |
| 해결 | 볼륨 >= `params_B × 2 × 1.1 GB` |

### E4. HF cache 중복 (hub/ 복사본)

| 항목 | 내용 |
|------|------|
| 증상 | 볼륨 급속 소진, `hub/` 디렉토리 하위에 복사본 |
| 트리거 | `cache_dir`과 `HUGGINGFACE_HUB_CACHE` 경로 불일치 |
| 해결 | `HF_HOME` + `HUGGINGFACE_HUB_CACHE` + `TRANSFORMERS_CACHE` 전부 동일 경로 |

---

## F. RunPod 인프라

### F1. Pod SSH 접속 불가

| 항목 | 내용 |
|------|------|
| 증상 | `Connection refused` 또는 `no route to host` |
| 트리거 | Spot pod 뺏김, image pull 미완료, SSH 키 누락 |
| 해결 | 3회 retry → terminate → 새 pod. 새 pod도 같은 호스트면 5분 대기 |

### F2. 이미지 pull 타임아웃

| 항목 | 내용 |
|------|------|
| 증상 | SSH 대기 60분+ 초과 |
| 트리거 | 대형 이미지(9 GB+) + 미캐시 머신 |
| 해결 | `cloudType: SECURE` 우선 → 소형 이미지 대체 |

### F3. Spot 인스턴스 중단

| 항목 | 내용 |
|------|------|
| 증상 | 실행 중 pod 예고 없이 terminate |
| 트리거 | `interruptible: true` pod + 장기 작업 |
| 해결 | 1시간+ 작업은 `interruptible: false` 강제 |

### F4. vCPU 초과 병렬 프로세스

| 항목 | 내용 |
|------|------|
| 증상 | CPU 100%, SSH 응답 없음 |
| 트리거 | 병렬 프로세스 > vCPU |
| 해결 | `MAX_PARALLEL = vCPU_count` 강제 |

### F5. 중복 pod 생성

| 항목 | 내용 |
|------|------|
| 증상 | 동일 실험 pod 여러 개, 비용 배수 |
| 트리거 | 기존 pod 확인 없이 신규 생성 |
| 해결 | Pre-deploy triage: 기존 pod 조회 → 재활용 판단 |

### F6. Network volume 실수 삭제

| 항목 | 내용 |
|------|------|
| 증상 | 수 GB~TB 모델 데이터 소멸 |
| 트리거 | 자동화 스크립트의 cleanup 단계 |
| 해결 | Hook에서 `deleteNetworkVolume` 명령 차단. 사용자 명시 승인 시에만 허용 |

### F7. API 인증 `api-key:` 헤더 무효

| 항목 | 내용 |
|------|------|
| 증상 | GraphQL `myself: null` |
| 트리거 | RunPod GraphQL은 Bearer token만 인식 |
| 해결 | `Authorization: Bearer $API_KEY` 헤더 |

### F8. `gpuCount` 누락 → SUPPLY_CONSTRAINT

| 항목 | 내용 |
|------|------|
| 증상 | Pod 생성 요청 시 에러 (실제 재고 있음) |
| 해결 | `podFindAndDeployOnDemand` mutation에 `gpuCount: 1` 필수 |

### F9. Idle pod 방치 과금

| 항목 | 내용 |
|------|------|
| 증상 | GPU 0% pod이 시간당 과금 계속 |
| 트리거 | 실험 완료 후 terminate 안 함 |
| 해결 | Idle monitor 자동화 (15분 주기, GPU 0% + 30분+ → stop) |
