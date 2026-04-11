#!/usr/bin/env python3
"""
Preflight Guard — 배포 전 Python 스크립트 정적 분석

Usage:
    python preflight-guard.py <script.py> [<script2.py> ...]

Exit codes:
    0 = 모든 검사 통과
    1 = 하나 이상 실패
    2 = 사용법 오류

검사 항목 (CHANGELOG.md의 실제 크래시에서 도출):
    1. Syntax 파싱
    2. total_memory 오타 탐지
    3. transformers 버전 호환성 (MODEL_ID와)
    4. RunPod 스크립트에 google.colab import 혼입
    5. Colab 스크립트에 /workspace/ 경로
    6. bitsandbytes 버전 핀 고정
    7. pip 패키지 충돌 검사 (optimum + transformers)
    8. VRAM 과부하 — 모델 크기 vs max_memory
    9. GPTQ 검증된 버전 강제
   10. RunPod 스크립트에 HF_HOME 설정
   11. output_hidden_states=True 대용량 모델 RAM 경고
   12. BnB 4-bit + max_memory 'cpu' 엔트리 (meta tensor 버그)
   13. BnB 대형 모델 + expandable_segments 누락
   14. Multi-GPU hook에서 GPU 텐서 직접 연산
"""

import sys
import re
import os
import ast


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


def check_source(source: str, fname: str, file_path: str = "") -> list:
    """소스 코드 검사 → 에러 목록 반환"""
    errors = []

    if not fname.endswith('.py'):
        return errors

    # hook/guard 스크립트 자체 제외
    if fname.endswith('-guard.py') or 'pipeline/' in file_path:
        return errors

    # 1. Syntax 검사
    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"syntax error: {e}")
        return errors

    # 2. `.total_mem` 오타 탐지
    typo_pattern = r'\.' + 'total_mem' + r'\b(?!ory)'
    for m in re.finditer(typo_pattern, source):
        line_start = source.rfind('\n', 0, m.start()) + 1
        line = source[line_start:source.find('\n', m.start())]
        stripped = line.lstrip()
        if not stripped.startswith('#'):
            errors.append("torch.cuda.get_device_properties(i).total_memory 사용 (오타 확인)")
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
                    errors.append(
                        f"transformers=={tf_ver} < required {min_ver} for {model_id}"
                    )
                elif ">=" in tf_op and specified < required:
                    errors.append(
                        f"transformers>={tf_ver} < required {min_ver} for {model_id}"
                    )
            except ImportError:
                if tf_ver < min_ver:
                    errors.append(
                        f"transformers {tf_op}{tf_ver} < required {min_ver} for {model_id}"
                    )

    # 4. RunPod 스크립트에 google.colab
    if 'runpod' in fname.lower() and re.search(r'from\s+google\.colab\s+import', source):
        errors.append("RunPod 스크립트에 google.colab import")

    # 5. Colab 스크립트에 /workspace/
    if 'colab' in fname.lower() or ('google.colab' in source and 'runpod' not in fname.lower()):
        if re.search(r'/workspace/', source) and '/content/' not in source:
            errors.append("Colab 스크립트에 /workspace/ 경로 (RunPod 전용)")

    # 6. bitsandbytes 버전 핀
    uses_bnb = (
        'BitsAndBytesConfig' in source
        or 'load_in_8bit' in source
        or 'load_in_4bit' in source
    )
    if uses_bnb:
        pip_lines = [
            line for line in source.split('\n')
            if 'pip' in line and 'install' in line and 'bitsandbytes' in line
        ]
        bnb_name = 'bitsand' + 'bytes'
        for pip_line in pip_lines:
            has_bnb_name = re.search(bnb_name + r'(?:["\']|,|\s|$)', pip_line)
            has_bnb_ver = re.search(bnb_name + r'[><=!]+[\d.]+', pip_line)
            if has_bnb_name and not has_bnb_ver:
                errors.append(
                    f"{bnb_name} 버전 미지정 — 고정 핀 필수"
                )
                break

    # 7. pip 패키지 충돌
    pip_all = ' '.join(
        line for line in source.split('\n')
        if 'pip' in line and 'install' in line
    )
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

    # 8. VRAM 과부하
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
                        f"GPU 할당 합계 {gpu_total}GiB — CPU offload 과다 위험"
                    )

    # 9. GPTQ 검증 버전
    if 'GPTQ' in source or 'auto-gptq' in source or 'auto_gptq' in source:
        pip_str = ' '.join(
            line for line in source.split('\n')
            if 'pip' in line and 'install' in line
        )
        opt_m = re.search(r'optimum[=!<>]+([\d.]+)', pip_str)
        gptq_m = re.search(r'auto.gptq[=!<>]+([\d.]+)', pip_str)
        if opt_m:
            try:
                from packaging.version import Version
                valid = [Version(v) for v in ('1.23.0', '1.23.1', '1.23.2', '1.23.3')]
                if Version(opt_m.group(1)) not in valid:
                    errors.append(
                        f"GPTQ: optimum=={opt_m.group(1)} 미검증 — "
                        f"검증 조합: transformers==4.51.3 + optimum==1.23.3 + auto-gptq==0.7.1"
                    )
            except ImportError:
                pass
        if gptq_m and gptq_m.group(1) != '0.7.1':
            errors.append(
                f"GPTQ: auto-gptq=={gptq_m.group(1)} 미검증 — 검증 버전: 0.7.1"
            )

    # 10. RunPod 스크립트에 HF_HOME
    if 'runpod' in fname.lower():
        if 'HF_HOME' not in source:
            errors.append(
                "RunPod 스크립트에 HF_HOME 미설정 — "
                "os.environ['HF_HOME'] = '/workspace/.cache_hf' 필수"
            )

    # 11. output_hidden_states + 대형 모델
    code_lines = [l for l in source.split('\n') if not l.strip().startswith('#')]
    if 'output_hidden_states=True' in '\n'.join(code_lines) and model_match:
        model_key = model_match.group(1)
        model_gb = MODEL_VRAM_GB.get(model_key, 0)
        if model_gb >= 70:
            errors.append(
                f"메모리 위험: {model_key} ({model_gb}GB) + output_hidden_states=True → "
                f"전체 레이어 hidden states 동시 보관, RAM 폭발. "
                f"register_forward_hook으로 레이어별 순차 처리 필수"
            )

    # 12. BnB 4-bit + max_memory 'cpu' 엔트리 (meta tensor 버그)
    if uses_bnb and 'load_in_4bit' in source:
        has_cpu_in_maxmem = re.search(r'["\']cpu["\']\s*:\s*["\']?\d+', source)
        has_auto_device_map = re.search(r'device_map\s*=\s*["\']auto["\']', source)
        if has_cpu_in_maxmem and has_auto_device_map:
            errors.append(
                "BnB 4-bit + max_memory 'cpu' 키 + device_map='auto' 조합 금지. "
                "accelerate가 BF16 크기 기준 CPU offload → meta tensor 버그. "
                "해결: 'cpu' 엔트리 제거 + 수동 device_map"
            )

    # 13. BnB 대형 모델 + expandable_segments 누락
    if uses_bnb and 'load_in_4bit' in source and model_match:
        model_key = model_match.group(1)
        model_gb = MODEL_VRAM_GB.get(model_key, 0)
        if model_gb >= 100:
            has_expandable = re.search(
                r'PYTORCH_CUDA_ALLOC_CONF.*expandable_segments\s*:\s*True',
                source,
            )
            if not has_expandable:
                errors.append(
                    f"대형 모델 ({model_key}, ~{model_gb}GB) BnB 4-bit 로딩 시 "
                    f"expandable_segments 미설정. "
                    f"해결: os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'"
                )

    # 14. Multi-GPU hook GPU 텐서 직접 연산
    if 'register_forward_hook' in source and 'device_map' in source:
        hook_patterns = re.findall(
            r'def\s+_hook.*?(?=\n    def |\nclass |\n[a-zA-Z])',
            source, re.DOTALL,
        )
        for hook_body in hook_patterns:
            has_gpu_arith = re.search(
                r'(out\[0\].*?\.float\(\).*?-|h_out.*?\.float\(\).*?h_in.*?\.float\(\))',
                hook_body,
            )
            has_cpu_first = '.cpu().float()' in hook_body or '.detach().cpu()' in hook_body
            if has_gpu_arith and not has_cpu_first:
                errors.append(
                    "Multi-GPU hook에서 텐서 직접 연산 → device mismatch 위험. "
                    "해결: .detach().cpu().float() 먼저 호출"
                )
                break

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python preflight-guard.py <script.py> [<script2.py> ...]", file=sys.stderr)
        sys.exit(2)

    total_failed = 0
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"[SKIP] {path}: 파일 없음", file=sys.stderr)
            continue
        try:
            source = open(path).read()
        except Exception as e:
            print(f"[ERROR] {path}: 읽기 실패 ({e})", file=sys.stderr)
            total_failed += 1
            continue

        fname = os.path.basename(path)
        errors = check_source(source, fname, path)
        if errors:
            print(f"[FAIL] {path}")
            for err in errors:
                print(f"  - {err}")
            total_failed += 1
        else:
            print(f"[OK]   {path}")

    if total_failed:
        print(f"\n총 {total_failed}개 파일 실패", file=sys.stderr)
        sys.exit(1)

    print("\n모든 검사 통과")
    sys.exit(0)


if __name__ == "__main__":
    main()
