# RunPod Deploy Agent

**Claude Code subagent + 자동 파이프라인**으로 RunPod 대형 모델 배포 시
크래시와 자원 낭비를 사전에 차단한다.

---

## 무엇을 제공하는가

### 1. Claude Code Subagent (`agents/runpod-deploy.md`)
Claude Code에서 `/agent runpod-deploy` 로 호출 가능한 전용 에이전트.
배포 계획 수립부터 모니터링, cleanup까지 전 과정 자동화.

### 2. PreToolUse Hook (`hooks/deploy-preflight-guard.py`)
Write / Edit 도구 호출 시 자동으로 Python 스크립트 정적 분석.
크래시 유발 패턴 14종 탐지 → block.

### 3. Runpod Bash Guard (`hooks/runpod-guard.js`)
메인 Claude Code 세션에서 직접 RunPod 명령 실행 차단.
반드시 전용 agent 경유하도록 강제.

### 4. 배포 파이프라인 (`tools/`)
- `preflight-guard.py` — 스크립트 정적 분석 CLI
- `pod-deploy-guard.py` — 안전한 pod 생성 CLI
- `idle-monitor-guard.py` — 유휴 pod 자동 감지 데몬
- `templates/` — 재사용 가능한 안전 패턴 (env setup, manual device_map, loader)

### 5. 규칙 & 문서 (`rules/`, `docs/`)
- 14개 Core Rules (R0~R14)
- Crash Catalog (증상별 색인)
- BnB 4-bit 안전 레시피
- RunPod API 특이사항

---

## 누가 써야 하는가

- RunPod에서 **70B+ 대형 모델**을 BnB 4-bit로 로딩하는 ML 엔지니어
- Claude Code로 LLM 작업을 자동화하는 개발자
- 이전에 device_map OOM, meta tensor 크래시, spot instance 중단 등으로 고생한 적 있는 사람

---

## 설치

### 옵션 A: 개별 파일 수동 설치

```bash
git clone https://github.com/popixoxipop-collab/RunPod_Deploy_Agent.git
cd RunPod_Deploy_Agent
bash install/install.sh
```

install.sh 가 하는 일:
1. `agents/runpod-deploy.md` → `~/.claude/agents/runpod-deploy.md`
2. `hooks/*.py`, `hooks/*.js` → `~/.claude/hooks/scripts/`
3. `tools/` → `~/.runpod-deploy-agent/tools/`
4. Claude Code settings에 hook 등록 안내 출력

### 옵션 B: Claude Code Plugin (향후 지원 예정)

```bash
claude plugin add popixoxipop-collab/RunPod_Deploy_Agent
```

---

## 사용법

### Claude Code에서 에이전트 호출

```
/agent runpod-deploy

대형 모델을 BnB 4-bit로 로딩하는 스크립트를 작성하고
A100 × N pod에 배포해줘.
```

에이전트가 자동으로:
1. 스크립트 정적 분석 (14종 크래시 패턴 검사)
2. 메모리 예산 계산 (모델 크기 vs GPU VRAM)
3. 기존 pod 조회 (중복 생성 방지)
4. Pod 생성 (interruptible=false 강제, PUBLIC_KEY 체크)
5. 실행 후 60초 double-check
6. 모니터링 (크래시 시 로그 회수 → terminate)

### CLI 단독 사용

```bash
# 1. 스크립트 정적 분석
python tools/preflight-guard.py your_script.py

# 2. 기존 pod 조회
export RUNPOD_API_KEY=$(cat ~/.runpod_api_key)
python tools/pod-deploy-guard.py --list

# 3. 안전한 pod 생성
python tools/pod-deploy-guard.py \
    --gpu-type "NVIDIA A100-SXM4-80GB" \
    --gpu-count N \
    --volume-id YOUR_VOL_ID \
    --datacenter US-MD-1 \
    --public-key "$(cat ~/.ssh/id_ed25519.pub)" \
    --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 4. 유휴 감지 (백그라운드 데몬)
nohup python tools/idle-monitor-guard.py --action stop --poll-interval-sec 900 &
```

---

## 파일 구조

```
RunPod_Deploy_Agent/
├── README.md                        ── 이 파일
├── LICENSE
├── agents/
│   └── runpod-deploy.md            ── Claude Code subagent 정의
├── hooks/
│   ├── deploy-preflight-guard.py   ── PreToolUse: Write/Edit 정적 분석
│   ├── runpod-guard.js             ── PreToolUse: Bash, 메인 세션 차단
│   └── commit-guard.js             ── (선택) PreToolUse: 커밋 메시지 검증
├── tools/
│   ├── preflight-guard.py          ── 정적 분석 CLI
│   ├── pod-deploy-guard.py         ── Pod 생성 CLI
│   ├── idle-monitor-guard.py       ── 유휴 감지 데몬
│   ├── requirements.txt
│   └── templates/
│       ├── env_setup.py            ── 필수 환경변수
│       ├── bnb_manual_device_map.py ── 수동 device_map 유틸
│       └── large-model-loader-guard.py ── 대형 모델 로더
├── rules/
│   ├── core-rules.md               ── R0~R14
│   ├── crash-catalog.md            ── 증상별 색인
│   ├── bnb-4bit-recipe.md          ── BnB 4-bit 안전 레시피
│   └── api-quirks.md               ── RunPod API 특이사항
├── examples/
│   ├── load_70b_example.py         ── 70B 로딩 예제
│   └── invoke_agent.md             ── Claude Code 호출 예제
└── install/
    └── install.sh                  ── ~/.claude/ 에 설치
```

---

## 왜 필요한가

주요 크래시 유형:

| 크래시 | 원인 |
|--------|------|
| GPU 단편화 OOM | `expandable_segments` 미설정 |
| BnB meta tensor | `max_memory`에 `"cpu"` 엔트리 |
| Spot instance 회수 | 장기 작업에 spot |
| Network volume 삭제 사고 | 자동화 cleanup 미승인 |
| 중복 pod 생성 | 기존 pod 확인 안 함 |
| Idle pod 방치 | 완료 후 terminate 안 함 |

대부분은 **사전 정적 분석만 했어도 막을 수 있던 것들**.
이 에이전트는 그 정적 분석을 자동화한다.

---

## 기여

새 크래시 겪으면:

1. `rules/crash-catalog.md` 에 증상/원인/해결 추가
2. `hooks/deploy-preflight-guard.py` 의 `check_source()` 에 탐지 규칙 추가
3. `examples/` 에 재현 가능한 최소 예제
4. `agents/runpod-deploy.md` 의 "Known Issues" 섹션 업데이트
5. PR 제출

---

## 라이선스

MIT
