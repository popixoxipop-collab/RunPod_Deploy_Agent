#!/usr/bin/env python3
"""
Code Quality Preflight Guard (PreToolUse: Write, Edit, Bash)

모든 Python 코드 작성/수정/배포 시 기본 품질 검증을 강제한다.

탐지 시점:
  - Write: 새 .py 파일 작성
  - Edit: 기존 .py 파일 수정
  - Bash: jupytext 변환, scp/rsync 업로드

검사 항목:
  1. syntax 파싱
  2. total_mem 오타 → total_memory
  3. transformers 버전 호환성 (MODEL_ID ↔ 요구 버전)
  4. is_torch_fx_available 참조 (transformers 5.x에서 제거)
  5. RunPod 스크립트에 google.colab import 혼입
  6. Colab 스크립트에 /workspace/ 경로
  7. bitsandbytes SCB 패치 + 버전 검사
  8. pip 패키지 충돌 검사 (optimum + transformers)
  9. VRAM 과부하 — 모델 크기 vs max_memory 설정
 10. GPTQ 사용 시 검증된 버전 조합 강제
 11. RunPod 스크립트에 HF_HOME 환경변수 누락
 12. output_hidden_states=True 대용량 모델 메모리 경고

[시행착오 기록 — 2026-04-09 Qwen3-235B GPTQ]
- 4-bit bitsandbytes: CPU offload 불가 → 8-bit 또는 GPTQ 사용
- GPTQ 검증 버전: transformers==4.51.3 + optimum==1.23.3 + auto-gptq==0.7.1
- H100 NVL x1 = 96GB VRAM, GPTQ-Int4 235B = 120GB → CPU offload 필수
- output_hidden_states=True 로 94레이어 전부 올리면 RAM 폭발 → forward hook 사용
- HF_HOME 미설정 → 모델이 container disk(/root/.cache)에 쌓여 50GB 초과
- idle monitor IDLE_STRIKES=2 → 모델 로딩 중 GPU 0%로 오탐 종료 → IDLE_STRIKES=10
- bitsandbytes SCB 버그: Colab Python3.12 + device_map=auto → 몽키패치 필수
- auto-gptq 0.7.1 + PyTorch 2.4: rshift_cuda Half 에러 → qlinear_cuda_old.py
  line 296(qzeros)과 line 311(qweight)에 .to(torch.int32) 패치 필수
- auto-gptq CUDA ext 미설치 시 Triton 커널 컴파일에 CPU 2000%+ 무한 대기
  → I/O 거의 0 (모델 파일 안 읽음), GPU 0% 상태로 15분+ 멈춤
  → 해결: GPTQConfig(bits=4, disable_exllama=True) + DISABLE_EXLLAMA=1 환경변수
  → CUDA old backend(qlinear_cuda_old) 직접 사용으로 Triton 컴파일 우회
- PyTorch 2.11 + CUDA 13.0 환경에서 auto-gptq 0.7.1 CUDA ext pre-built wheel 없음
  → pip install --no-deps로 설치 시 wheel만 설치되어 CUDA ext 빠짐
  → BUILD_CUDA_EXT=1 --no-build-isolation --no-deps --force-reinstall로 소스 빌드 필요
  → 단, --force-reinstall이 torch/transformers 버전 덮어쓰므로 반드시 --no-deps 병행
- HF cache 중복: snapshot_download(cache_dir=X)와 from_pretrained(MODEL_ID)가
  hub/ 하위에 복사본 생성 → 200GB 볼륨 초과
  → 해결: HUGGINGFACE_HUB_CACHE=cache_dir로 통일, 또는 LOCAL_PATH 직접 지정

[시행착오 기록 — 2026-04-10 Qwen3-235B BnB 4-bit]
- ★ BnB 사용 시 볼륨 용량 선행 계산 필수 ★
  BnB 4-bit는 bf16 원본을 다운로드 후 로딩 시 양자화 → 디스크에 bf16 전체 필요
  공식: 볼륨 >= 모델 파라미터수(B) × 2 × 1.1 (GB, 10% 여유)
  예: 235B → 470GB × 1.1 = 517GB → 최소 600GB 볼륨
  예: 671B(R1) → 1,342GB × 1.1 = 1,476GB → 최소 1,500GB 볼륨 (fp8이면 671GB)
  GPTQ 프리 양자화는 디스크 작지만 라이브러리 호환 지옥 → BnB가 안정적
- GPTQ vs BnB 판단 기준:
  볼륨 여유 있으면 → BnB (단순, 안정, 72B에서 검증)
  볼륨 부족하면 → GPTQ (디스크 절약이나 auto-gptq 호환성 문제 각오)
- transformers>=4.51.0 + PyTorch 2.4: set_submodule 없음
  → PyTorch 2.5+ 필요 (set_submodule은 PyTorch 2.5에서 추가)
  → 또는 transformers<4.48 사용 (set_submodule 미사용 버전)
- PyTorch 업그레이드 시 반드시 torchvision도 함께 업그레이드
  → torch만 올리면 torchvision::nms operator 에러
  → pip install torch torchvision --upgrade 한 번에
- bitsandbytes 0.49+ + PyTorch 2.11(cu130): libnvJitLink.so.13 못 찾음
  → LD_LIBRARY_PATH에 /usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib 추가 필수
  → 또는 nvidia-nvjitlink-cu13 패키지 설치
- 스크립트 pip install에 BnB 버전 핀 지정: bitsandbytes>=0.44.0
  → 최신(0.49.2)은 PyTorch 2.11과 호환되나 LD_LIBRARY_PATH 필요
- RunPod NFS 대용량 다운로드 I/O error (OS error 5) 빈발
  → 500GB+ 다운로드 시 "IO Error: Input/output error (os error 5)" 랜덤 발생
  → xet 프로토콜이 NFS와 호환 불량 → HF_HUB_DISABLE_XET=1 환경변수 필수
  → max_workers=2로 제한하여 동시 쓰기 줄이기
  → try/except retry loop 5회 감싸기 (while downloading - for attempt in range(5))
  → snapshot_download은 incomplete 파일 resume 지원하므로 재시도 안전
- DeepSeek-R1 fp8 네이티브 → BnB 4-bit 로딩 시 torch_dtype 강제 필요
  → BnB 4-bit은 torch.float8_e4m3fn 입력 지원 안 함 (fp16/fp32만 받음)
  → 에러: "Blockwise 4bit quantization only supports 16/32-bit floats, but got torch.float8_e4m3fn"
  → from_pretrained에 torch_dtype=torch.float16 반드시 명시
  → 동시에 config에서 fp8 quantization_config 제거 필수 (del _cfg.quantization_config)
- BnB 4-bit + MoE 대형 모델(R1 671B) CPU 오프로드 에러
  → "Some modules are dispatched on the CPU or the disk"
  → accelerate가 bf16 크기(1342GB) 기준으로 계산해서 GPU 용량 부족 판단
  → 해결: BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
  → max_memory에 "cpu": "500GiB" 추가 (CPU 여유 공간 명시)
  → GPU당 메모리는 A100 80GB 기준 "76GiB"까지 (2GB 여유)

[시행착오 기록 — 대형 BnB 4-bit 로딩 ★ 중요 ★]
- ★★ BnB 4-bit 대형 모델 로딩 시 max_memory에 "cpu" 키 절대 금지 ★★
  → "cpu" 엔트리가 있으면 accelerate가 bf16 원본 크기 기준 계산
  → GPU 예산 < bf16 크기 → 일부 레이어 "cpu" 자동 배치
  → 로딩 완료 후 첫 model call 시 accelerate AlignDevicesHook.pre_forward가
     offload=True 모드로 set_module_tensor_to_device 호출
  → BnB 0.49.2의 Params4bit.to() → quant_state.to() → self.code.to(device)
  → self.code이 meta tensor (init_empty_weights 컨텍스트에서 materialize 안 됨)
  → NotImplementedError: Cannot copy out of meta tensor; no data!
  → 해결: "cpu" 엔트리 제거 + 수동 device_map으로 모든 레이어를 GPU ID에 명시 할당
- ★★ 수동 device_map 패턴 (device_map="auto" 대신) ★★
  ```python
  _cfg = AutoConfig.from_pretrained(LOCAL_PATH, trust_remote_code=True)
  n_layers = _cfg.num_hidden_layers
  n_gpus = torch.cuda.device_count()
  layers_per_gpu = [(n_layers + i) // n_gpus for i in range(n_gpus)]
  layers_per_gpu.reverse()  # 앞쪽 GPU(embed 포함)에 더 많이
  device_map = {"model.embed_tokens": 0}
  idx = 0
  for gpu_id, n in enumerate(layers_per_gpu):
      for _ in range(n):
          device_map[f"model.layers.{idx}"] = gpu_id
          idx += 1
  device_map["model.norm"] = n_gpus - 1
  device_map["lm_head"] = n_gpus - 1
  # device_map.values()에 "cpu", "disk" 없음 → offload=False 훅 → 버그 경로 차단
  ```
- ★★ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 필수 ★★
  → BnB on-the-fly 양자화 시 BF16 shard 반복 alloc/free → 메모리 단편화
  → 대형 모델 로딩 중 N GiB "reserved but unallocated" → OOM
  → 해결: 스크립트 최상단 (torch import 전)에
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  → PyTorch 2.1+ 기능. 가변 크기 segment로 파편화 대폭 감소
- ★★ BnB 4-bit hook에서 device mismatch 주의 ★★
  → multi-GPU + device_map에서 register_forward_hook의 out/inp이 다른 GPU에 있을 수 있음
  → h_out.float() - h_in.float() 직접 연산하면 "Expected all tensors to be on the same device"
  → 해결: .detach().cpu().float() 먼저 해서 CPU에서 연산
- BnB 4-bit 옵션 권장:
  → bnb_4bit_quant_type="nf4"
  → bnb_4bit_use_double_quant=False (double_quant도 meta tensor 이슈 유발 가능)
- RunPod Spot instance 금지 (1시간+ 작업):
  → interruptible=true 는 예고 없이 뺏길 수 있음
  → 장시간 로딩을 spot에서 돌리면 자살행위
  → 반드시 interruptible=false (on-demand) 사용
- torch._inductor 초기 컴파일 대기:
  → custom modeling 파일 사용 모델이 torch.compile 사용 시
  → 프로세스 시작 후 ~5-10분 동안 inductor compile_worker 구동
  → 이 동안 GPU 0%, tqdm 출력 없음 → 멈춘 것처럼 보임 (정상)
- NFS 페이지 캐시 보존:
  → 같은 볼륨에서 pod 재생성해도 NFS 서버 페이지 캐시는 유지
  → 재시도 시 이전 로딩 덕분에 25s/shard (cold 60-120s/shard 대비 3-5배)
- 반응형 디버깅 금지 (사전 점검 우선):
  → 3시간+ 실험은 시작 전 스크립트 전체 정적 분석 필수
  → 크래시 → 수정 → 재시도 반복 = credit + 시간 소진
  → 사전 체크리스트: max_memory, device_map, expandable_segments, hook device, bnb 옵션

[대용량 다운로드 기본 정책 — 500GB+ 모델]
- ★ hf_transfer 필수 ★ (Rust 기반 병렬 다운로더, HF 공식)
  → pip install hf_transfer
  → 환경변수: HF_HUB_ENABLE_HF_TRANSFER=1
  → max_workers=8 (기본 2보다 4배 빠름)
  → 실측: 일반 다운로드 38MB/s → hf_transfer 770MB/s (20배)
- 다운로드 기본 셋업 체크리스트:
  1. pip install hf_transfer
  2. HF_HUB_ENABLE_HF_TRANSFER=1
  3. snapshot_download(..., max_workers=8)
  4. retry loop 10회 (I/O error 대비)
  5. HF_HOME + HUGGINGFACE_HUB_CACHE 동일 경로 설정 (hub/ 중복 방지)
- 언제 xet 비활성화 (HF_HUB_DISABLE_XET=1):
  → NFS에서 "IO Error: Input/output error (os error 5)" 발생 시에만
  → 기본은 hf_transfer만 사용 (xet와 별개)
  → 해결: from_pretrained에 LOCAL_PATH(스냅샷 직접 경로) 전달, hub/ 생성 방지
"""
import json, sys, re, os, ast


# 모델별 최소 transformers 요구 버전
# (사용자 프로젝트에서 직접 확장 가능)
KNOWN_MIN_VERSIONS = {
    "Qwen/Qwen3-235B-A22B": "4.51.0",
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4": "4.51.0",
    "deepseek-ai/DeepSeek-R1": "4.46.3",
    "deepseek-ai/DeepSeek-V3": "4.46.3",
    "meta-llama/Llama-3.1-70B-Instruct": "4.43.0",
    "meta-llama/Llama-3.1-405B": "4.43.0",
}

# 알려진 모델 크기 (GB)
MODEL_VRAM_GB = {
    "Qwen/Qwen3-235B-A22B-GPTQ-Int4": 120,
    "Qwen/Qwen3-235B-A22B": 470,
    "deepseek-ai/DeepSeek-R1": 671,
    "deepseek-ai/DeepSeek-V3": 671,
    "meta-llama/Llama-3.1-70B-Instruct": 140,
    "meta-llama/Llama-3.1-405B": 810,
}


def check_source(source, fname, file_path=""):
    """소스 코드를 검사하여 에러 목록 반환"""
    errors = []

    # hook 스크립트 자체 제외
    if fname.endswith('-guard.py') or fname.endswith('-guard.js') or 'hooks/scripts' in file_path:
        return errors

    # .py 파일만
    if not fname.endswith('.py'):
        return errors

    # 1. Syntax 검사
    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"syntax error: {e}")
        return errors

    # 2. total_mem 오타
    for m in re.finditer(r'\.total_mem\b(?!ory)', source):
        line_start = source.rfind('\n', 0, m.start()) + 1
        line = source[line_start:source.find('\n', m.start())]
        stripped = line.lstrip()
        if not stripped.startswith('#') and not stripped.startswith('"') and not stripped.startswith("'"):
            errors.append("'.total_mem' 오타 → '.total_memory'로 수정 필요")
            break

    # 3. transformers 버전 호환성
    model_match = re.search(r'MODEL_ID\s*=\s*["\']([^"\']+)["\']', source)
    tf_match = re.search(r'transformers([><=!]+)([\d.]+)', source)

    if model_match and tf_match:
        model_id = model_match.group(1)
        tf_op = tf_match.group(1)
        tf_ver = tf_match.group(2)
        min_ver = KNOWN_MIN_VERSIONS.get(model_id)
        if min_ver:
            try:
                from packaging.version import Version
                specified = Version(tf_ver)
                required = Version(min_ver)
                if "==" in tf_op and specified < required:
                    errors.append(f"transformers=={tf_ver} 지정 but {model_id}은 >={min_ver} 필요")
                elif ">=" in tf_op and specified < required:
                    errors.append(f"transformers>={tf_ver} 지정 but {model_id}은 >={min_ver} 필요")
            except ImportError:
                if tf_ver < min_ver:
                    errors.append(f"transformers {tf_op}{tf_ver} 지정 but {model_id}은 >={min_ver} 필요")

    # 4. is_torch_fx_available
    for m in re.finditer(r'is_torch_fx_available', source):
        line_start = source.rfind('\n', 0, m.start()) + 1
        line = source[line_start:source.find('\n', m.start())]
        if not line.lstrip().startswith('#'):
            errors.append("is_torch_fx_available 참조 — transformers 5.x에서 제거됨")
            break

    # 5. RunPod 스크립트에 google.colab
    if 'runpod' in fname.lower() and re.search(r'from\s+google\.colab\s+import', source):
        errors.append("RunPod 스크립트에 google.colab import 포함")

    # 6. Colab 스크립트에 /workspace/ 경로
    if 'colab' in fname.lower() or (
        'google.colab' in source and 'runpod' not in fname.lower()
    ):
        if re.search(r'/workspace/', source) and '/content/' not in source:
            errors.append("Colab 스크립트에 /workspace/ 경로 사용 (RunPod 전용)")

    # 7. bitsandbytes SCB 패치 + 버전 검사
    uses_bnb = ('BitsAndBytesConfig' in source or 'load_in_8bit' in source
                or 'load_in_4bit' in source)
    if uses_bnb:
        # 7a. SCB 몽키패치 필수 (Colab Python 3.12 + device_map=auto)
        if '_safe_save' not in source and 'Linear8bitLt._save_to_state_dict' not in source:
            errors.append(
                "bitsandbytes SCB 몽키패치 누락 — "
                "BitsAndBytesConfig 사용 시 Linear8bitLt._save_to_state_dict 패치 필수 "
                "(Colab Python 3.12 + device_map='auto' SCB 버그)"
            )
        # 7b. pip install 버전 미지정
        pip_lines = [line for line in source.split('\n')
                     if 'pip' in line and 'install' in line and 'bitsandbytes' in line]
        if pip_lines:
            for pip_line in pip_lines:
                bnb_match = re.search(r'bitsandbytes(?:["\']|,|\s|$)', pip_line)
                bnb_ver_match = re.search(r'bitsandbytes[><=!]+[\d.]+', pip_line)
                if bnb_match and not bnb_ver_match:
                    errors.append("bitsandbytes 버전 미지정 — 'bitsandbytes>=0.44.0' 핀 고정 필수")
                    break

    # 8. pip 패키지 충돌 검사
    pip_all = ' '.join(line for line in source.split('\n')
                       if 'pip' in line and 'install' in line)
    if 'optimum' in pip_all and re.search(r'transformers[><=]*4\.(5[0-9]|[6-9]\d)', pip_all):
        opt_ver = re.search(r'optimum[=<>]+([\d.]+)', pip_all)
        if opt_ver:
            opt_v = opt_ver.group(1)
            try:
                from packaging.version import Version
                if Version(opt_v) < Version("1.23.0"):
                    errors.append(
                        f"optimum=={opt_v}은 transformers>=4.50과 충돌 — optimum>=1.23.0 필요"
                    )
            except ImportError:
                if opt_v < "1.23.0":
                    errors.append(f"optimum=={opt_v}은 transformers>=4.50과 충돌")

    # 9. VRAM 과부하 검사
    if model_match:
        model_key = model_match.group(1)
        model_gb = MODEL_VRAM_GB.get(model_key)
        if model_gb:
            gpu_mem_matches = re.findall(r'["\'](\d+)GiB["\']', source)
            if gpu_mem_matches:
                gpu_total = sum(int(x) for x in gpu_mem_matches if int(x) < 200)
                if gpu_total < model_gb * 0.5:
                    errors.append(
                        f"VRAM 경고: {model_key} ~{model_gb}GB, "
                        f"GPU 할당 합계 {gpu_total}GiB — CPU offload 과다로 매우 느릴 수 있음"
                    )

    # 10. GPTQ 검증된 버전 강제
    if 'GPTQ' in source or 'auto-gptq' in source or 'auto_gptq' in source:
        pip_str = ' '.join(l for l in source.split('\n') if 'pip' in l and 'install' in l)
        opt_m = re.search(r'optimum[=!<>]+([\d.]+)', pip_str)
        gptq_m = re.search(r'auto.gptq[=!<>]+([\d.]+)', pip_str)
        if opt_m:
            try:
                from packaging.version import Version
                if Version(opt_m.group(1)) not in [Version(v) for v in ('1.23.0','1.23.1','1.23.2','1.23.3')]:
                    errors.append(
                        f"GPTQ 버전 경고: optimum=={opt_m.group(1)} 미검증 — "
                        f"검증된 조합: transformers==4.51.3 + optimum==1.23.3 + auto-gptq==0.7.1"
                    )
            except ImportError:
                pass
        if gptq_m and gptq_m.group(1) != '0.7.1':
            errors.append(
                f"GPTQ 버전 경고: auto-gptq=={gptq_m.group(1)} 미검증 — 검증된 버전: 0.7.1"
            )

    # 11. RunPod 스크립트에 HF_HOME 누락
    if 'runpod' in fname.lower():
        if 'HF_HOME' not in source:
            errors.append(
                "RunPod 스크립트에 HF_HOME 미설정 — "
                "os.environ['HF_HOME'] = '/workspace/.cache_hf' 필수 "
                "(미설정 시 container disk /root/.cache에 쌓여 50GB 초과)"
            )

    # 12. output_hidden_states=True + 대용량 모델
    code_lines = [l for l in source.split('\n') if not l.strip().startswith('#')]
    if 'output_hidden_states=True' in '\n'.join(code_lines) and model_match:
        model_key = model_match.group(1)
        model_gb = MODEL_VRAM_GB.get(model_key, 0)
        if model_gb >= 70:
            errors.append(
                f"메모리 위험: {model_key} ({model_gb}GB)에서 output_hidden_states=True — "
                f"전체 layer state 동시 보관으로 RAM 폭발 위험. "
                f"필요한 layer만 선별 수집하도록 변경 필요"
            )

    # 13. BnB 4-bit + max_memory "cpu" 엔트리 (meta tensor 버그)
    if uses_bnb and 'load_in_4bit' in source:
        # max_memory에 "cpu" 키 있는지 + device_map="auto" 같이 쓰는지
        has_cpu_in_maxmem = re.search(r'["\']cpu["\']\s*:\s*["\']?\d+', source)
        has_auto_device_map = re.search(r'device_map\s*=\s*["\']auto["\']', source)
        if has_cpu_in_maxmem and has_auto_device_map:
            errors.append(
                "BnB 4-bit + max_memory에 'cpu' 키 + device_map='auto' 조합은 금지. "
                "accelerate가 BF16 크기 기준 CPU offload → BnB 0.49.2의 "
                "quant_state.code meta tensor 버그로 첫 model call 크래시. "
                "해결: 'cpu' 엔트리 제거 + 수동 device_map 구성"
            )

    # 14. BnB 대형 모델 + expandable_segments 누락 (2026-04-10 GPU5 단편화 OOM)
    if uses_bnb and 'load_in_4bit' in source and model_match:
        model_key = model_match.group(1)
        model_gb = MODEL_VRAM_GB.get(model_key, 0)
        if model_gb >= 100:  # 100GB+ 모델에서 on-the-fly 양자화
            has_expandable = re.search(
                r'PYTORCH_CUDA_ALLOC_CONF.*expandable_segments\s*:\s*True', source
            )
            if not has_expandable:
                errors.append(
                    f"대형 모델 ({model_key}, ~{model_gb}GB) BnB 4-bit 로딩 시 "
                    f"expandable_segments 미설정. "
                    f"해결: 스크립트 최상단에 "
                    f"os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' "
                    f"(BnB 반복 alloc/free로 메모리 단편화 → OOM)"
                )

    # 15. Multi-GPU hook에서 GPU 텐서 직접 연산 (device mismatch 위험)
    if 'register_forward_hook' in source and 'device_map' in source:
        # hook 함수 내에서 .float() - .float() 같은 패턴 (CPU로 안 내리고)
        hook_patterns = re.findall(
            r'def\s+_hook.*?(?=\n    def |\nclass |\n[a-zA-Z])',
            source, re.DOTALL
        )
        for hook_body in hook_patterns:
            # .float() 직접 연산 있는데 .cpu() 없는 경우
            has_gpu_arith = re.search(
                r'(out\[0\].*?\.float\(\).*?-|h_out.*?\.float\(\).*?h_in.*?\.float\(\))',
                hook_body
            )
            has_cpu_first = '.cpu().float()' in hook_body or '.detach().cpu()' in hook_body
            if has_gpu_arith and not has_cpu_first:
                errors.append(
                    "multi-GPU hook에서 텐서 직접 연산 감지 — device mismatch 위험. "
                    "accelerate가 layer 출력을 다음 GPU로 옮긴 후 훅이 발화할 수 있음. "
                    "해결: h_out = out[0].detach().cpu().float(); "
                    "h_in = inp[0].detach().cpu().float() 먼저 CPU로 내린 후 연산"
                )
                break

    return errors


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        if not file_path.endswith('.py') or not content:
            sys.exit(0)
        fname = os.path.basename(file_path)
        errors = check_source(content, fname, file_path)
        if errors:
            _block(fname, errors)
        else:
            _allow(fname)
        return

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "")
        new_string = tool_input.get("new_string", "")
        if not file_path.endswith('.py'):
            sys.exit(0)
        fname = os.path.basename(file_path)
        old_string = tool_input.get("old_string", "")
        if os.path.isfile(file_path):
            try:
                full_source = open(file_path).read()
                if old_string and new_string:
                    full_source = full_source.replace(old_string, new_string, 1)
                errors = check_source(full_source, fname, file_path)
            except Exception:
                sys.exit(0)
        else:
            errors = check_source(new_string, fname, file_path)
        if errors:
            _block(fname, errors)
        else:
            _allow(fname)
        return

    if tool_name == "Bash":
        command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""

        # 13-b. Network Volume 삭제 차단
        if 'deleteNetworkVolume' in command:
            _block("Network Volume 삭제 차단", [
                "Network Volume 삭제 금지 — 사용자 명시적 승인 없이 볼륨 삭제 불가. "
                "볼륨에 다운로드된 모델 데이터가 소멸되며 복구 불가능."
            ])

        # 13. RunPod pod 생성 시 PUBLIC_KEY 누락 검사
        if 'podFindAndDeployOnDemand' in command or 'podRuntimeAdd' in command:
            pod_errors = []
            if 'PUBLIC_KEY' not in command:
                pod_errors.append(
                    "RunPod pod 생성 시 PUBLIC_KEY env 누락 — SSH 접속 불가. "
                    "env 배열에 {key: \"PUBLIC_KEY\", value: \"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEyRDf2z9KZB8s6sZZ2xF0/2v6TPXzqQ5EcxhVRsqdB1 runpod-agi\"} 추가 필수"
                )
            if 'networkVolumeId' not in command and ('volumeInGb: 0' in command or 'volumeInGb:0' in command):
                pod_errors.append(
                    "RunPod pod 생성 시 networkVolumeId 누락 — "
                    "대형 모델 다운로드는 반드시 Network Volume 연결 필수 (ephemeral 볼륨은 pod 삭제 시 소멸)"
                )
            # 13-c. Spot instance 금지 (장기 작업) — 2026-04-11 R1 실험 손실 교훈
            if re.search(r'bidPerGpu|interruptible\s*:\s*true', command):
                pod_errors.append(
                    "RunPod spot instance(interruptible=true) 사용 감지 — "
                    "장기 작업(대형 모델 로딩, 3시간+ 실험)에 spot은 금지. "
                    "뺏기면 credit + 시간 동시 손실. 반드시 on-demand 사용 "
                    "(podFindAndDeployOnDemand + interruptible 제거 또는 false)"
                )
            if pod_errors:
                _block("RunPod pod 생성", pod_errors)
            # PUBLIC_KEY 있으면 통과
            sys.exit(0)

        if "jupytext" in command:
            jupytext_idx = command.find('jupytext')
            jupytext_part = command[jupytext_idx:]
            py_files = [f for f in re.findall(r'(\S+\.py)\b', jupytext_part)
                        if not f.startswith('-') and 'hooks/' not in f and 'guard' not in f]
            all_errors = []
            checked = []
            for py_file in py_files:
                if not os.path.isabs(py_file):
                    py_file = os.path.join(os.getcwd(), py_file)
                if not os.path.isfile(py_file):
                    continue
                try:
                    source = open(py_file).read()
                except Exception:
                    continue
                fname = os.path.basename(py_file)
                errors = check_source(source, fname, py_file)
                all_errors.extend(f"{fname}: {e}" for e in errors)
                checked.append(fname)
            if all_errors:
                _block(", ".join(checked), all_errors)
            elif checked:
                _allow(", ".join(checked))
            return

    sys.exit(0)


def _block(fname, errors):
    error_list = "\n".join(f"  • {e}" for e in errors)
    result = {
        "decision": "block",
        "reason": (
            f"[preflight-guard] 코드 품질 검증 실패!\n"
            f"  파일: {fname}\n"
            f"\n"
            f"{error_list}\n"
            f"\n"
            f"  ★ 수정 후 다시 시도하세요.\n"
            f"  ★ 모델 호환성: AutoConfig.from_pretrained(MODEL_ID).transformers_version 확인"
        )
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)


def _allow(fname):
    result = {
        "decision": "allow",
        "reason": f"[preflight-guard] 검증 통과: {fname}"
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
