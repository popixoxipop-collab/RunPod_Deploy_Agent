#!/usr/bin/env node
// RunPod 직접 실행 차단 hook (PreToolUse: Bash)
//
// 목적: RunPod 배포/모니터링/장애 대응은 전용 agent에 위임해야 함.
//       메인 Claude 세션에서 직접 Bash로 실행하려 하면 차단.
// Bypass: 에이전트 내부에서 명령 앞에 RUNPOD_AGENT=1 prefix 추가 시 통과.

let inputStr = '';
process.stdin.on('data', chunk => { inputStr += chunk; });
process.stdin.on('end', () => {
  const data = JSON.parse(inputStr || '{}');
  const toolInput = data.tool_input || data;
  const rawCommand = toolInput.command || '';
  const command = rawCommand.toLowerCase();

  // Bypass: RUNPOD_AGENT=1 prefix 있으면 즉시 통과
  if (/runpod_agent=1/i.test(rawCommand)) process.exit(0);

  // RunPod 배포 직접 실행 패턴
  const RUNPOD_DEPLOY_PATTERNS = [
    /python.*runpod.*\.py.*--submit/,
    /infra\.create_pod/,
    /infra\.wait_for_pod/,
    /podFindAndDeployOnDemand/,
    /podTerminate.*input.*podId/,
  ];

  // 장시간 모니터링 루프 (메인에서 pod 상태 polling)
  const RUNPOD_MONITOR_PATTERNS = [
    /sleep\s+\d{2,}.*runpod/,
    /tail.*runner\.log.*runpod/,
  ];

  // 직접 SSH/SCP/API 호출 (메인에서)
  const RUNPOD_SSH_PATTERNS = [
    /ssh\s+.*-p\s+\d{4,5}\s+root@/,
    /scp\s+.*-P\s+\d{4,5}\s+.*root@/,
    /rsync.*ssh\s+-p\s+\d{4,5}.*root@/,
    /curl.*api\.runpod\.io/,
  ];

  const allPatterns = [
    ...RUNPOD_DEPLOY_PATTERNS,
    ...RUNPOD_MONITOR_PATTERNS,
    ...RUNPOD_SSH_PATTERNS,
  ];
  const matched = allPatterns.find(p => p.test(command));

  if (matched) {
    console.log(JSON.stringify({
      decision: 'block',
      reason: [
        '🚫 RunPod 직접 실행 차단',
        '',
        '규칙: RunPod 배포/모니터링/장애 대응은 전용 agent에 위임해야 합니다.',
        '',
        '━━━ 올바른 위임 흐름 ━━━',
        '  메인 Claude → Agent tool (runpod-deploy agent) → 에이전트가 내부에서 실행',
        '',
        '━━━ Bypass 방법 (에이전트 내부) ━━━',
        '  모든 RunPod Bash 명령 앞에 RUNPOD_AGENT=1 prefix 추가:',
        '    RUNPOD_AGENT=1 ssh -p 12345 root@IP "..."',
        '    RUNPOD_AGENT=1 python tools/pod-deploy-guard.py --list',
        '  → RUNPOD_AGENT=1 있으면 이 hook 통과',
        '',
        '━━━ 도구 ━━━',
        '  /agent runpod-deploy         — 전용 에이전트 호출',
        '  tools/pod-deploy-guard.py    — 안전한 pod 생성 CLI',
        '  tools/preflight-guard.py     — 스크립트 정적 분석',
        '  tools/idle-monitor-guard.py  — 유휴 pod 감지',
        '',
        `차단된 명령: ${rawCommand.slice(0, 120)}`,
      ].join('\n'),
    }));
  } else {
    process.exit(0);
  }
});
