#!/usr/bin/env node
// RunPod Python 실험 실행 시 triton >= 3.4.0 설치 강제 hook (PreToolUse Bash)
//
// ★★★ 2026-04-13 사고 기록 — 절대 반복하지 말 것 ★★★
//
// 사고 1 — triton 누락 → BF16 폴백 → OOM
//   Pod2(jgdple91ixcoju)에서 triton 없음 → transformers가 MXFP4 대신 BF16 자동 폴백
//   → 120B BF16 = 240GB VRAM 필요 → 80GB H100 OOM.
//   원인: torch 2.4.1+cu124 환경에서 triton 3.4.0 pip install이 조용히 실패.
//   이 hook이 없었다면 계속 반복됐을 것.
//
// 사고 2 — 좀비 GPU 프로세스 → 새 프로세스까지 OOM
//   pkill 후에도 이전 프로세스(61.75GB)가 GPU에 잔존.
//   새 프로세스가 모델 로드하면서 합산 초과 → OOM.
//   교훈: 재시작 전 반드시 nvidia-smi 로 VRAM = ~1MB 확인 후 시작.
//
// ━━━ 올바른 RunPod 실험 재시작 순서 (이 순서를 반드시 지킬 것) ━━━
//   1. kill -9 $(pgrep -f eval_chimera) && sleep 5
//   2. nvidia-smi --query-gpu=memory.used --format=csv,noheader  → "1 MiB" 확인
//   3. python3 -c 'import triton; print(triton.__version__)'     → 3.4.0+ 확인
//   4. nohup python3 /workspace/scripts/eval_chimera.py ... &
//
// 탐지 조건: SSH로 RunPod에 접속해 Python 스크립트 실행 + triton 확인 없음
// 차단 조건: 명령 내에 triton 설치/확인 구문 없는 경우
// 통과 조건: pip install triton / python3 -c 'import triton' / TRITON_CHECKED=1

let inputStr = '';
process.stdin.on('data', chunk => { inputStr += chunk; });
process.stdin.on('end', () => {
  const data = JSON.parse(inputStr || '{}');
  const rawCommand = (data.tool_input || data).command || '';
  const command = rawCommand.toLowerCase();

  // RunPod SSH 명령 탐지 (ssh root@IP -p PORT ... 패턴)
  const isRunpodSsh = /ssh\s+.*root@\d+\.\d+\.\d+\.\d+.*-p\s+\d{4,5}/.test(command)
                   || /ssh\s+.*-p\s+\d{4,5}.*root@/.test(command);

  if (!isRunpodSsh) {
    process.exit(0);
  }

  // Python ML 스크립트 실행 탐지
  const isPythonScript = /python3?\s+.*\.(py)/.test(command)
                       || /nohup\s+python/.test(command);

  if (!isPythonScript) {
    process.exit(0);
  }

  // triton 설치/확인 구문 포함 여부
  const hasTritonInstall = /pip\s+install.*triton/.test(command)
                         || /triton_checked=1/i.test(command)
                         || /import\s+triton/.test(command);

  if (hasTritonInstall) {
    process.exit(0);
  }

  // triton 미확인 상태로 Python 스크립트 실행 시도 → 차단
  console.log(JSON.stringify({
    decision: 'block',
    reason: [
      '🚫 RunPod Python 실행 차단 — triton >= 3.4.0 미설치 확인',
      '',
      '이유: triton 없으면 MXFP4 → BF16 자동 폴백 → 80GB H100에서 OOM 발생',
      '      (2026-04-13 Pod2 OOM 재발 방지)',
      '',
      '━━━ 올바른 실행 방법 ━━━',
      '  SSH 명령에 triton 설치 구문을 포함하거나 TRITON_CHECKED=1을 앞에 붙일 것:',
      '',
      '  방법 1 — triton 설치 후 실행:',
      '    ssh root@IP -p PORT "pip install -q triton>=3.4.0 && nohup python3 ..."',
      '',
      '  방법 2 — 이미 설치된 경우 확인 후 실행:',
      '    ssh root@IP -p PORT "python3 -c \'import triton\' && nohup python3 ..."',
      '',
      '  방법 3 — 확인 완료 표시 (triton 설치됨을 확인한 경우):',
      '    ssh root@IP -p PORT "TRITON_CHECKED=1 nohup python3 ..."',
      '',
      `차단된 명령: ${rawCommand.slice(0, 120)}`,
    ].join('\n'),
  }));
});
