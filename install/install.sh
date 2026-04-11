#!/usr/bin/env bash
# RunPod Deploy Agent 설치 스크립트
#
# 설치 항목:
#   1. Claude Code 에이전트: agents/runpod-deploy.md → ~/.claude/agents/
#   2. Hook 스크립트: hooks/*.py, hooks/*.js → ~/.claude/hooks/scripts/
#   3. 파이프라인 도구: tools/ → ~/.runpod-deploy-agent/tools/
#
# 설치 후 수동 단계:
#   - ~/.claude/settings.json 에 hook 등록 (install.sh가 가이드 출력)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLAUDE_DIR="${CLAUDE_DIR:-$HOME/.claude}"
CLAUDE_AGENTS="$CLAUDE_DIR/agents"
CLAUDE_HOOKS="$CLAUDE_DIR/hooks/scripts"
RUNPOD_AGENT_HOME="${RUNPOD_AGENT_HOME:-$HOME/.runpod-deploy-agent}"

mkdir -p "$CLAUDE_AGENTS" "$CLAUDE_HOOKS" "$RUNPOD_AGENT_HOME"

echo "=== RunPod Deploy Agent 설치 ==="
echo "Repo:   $REPO_DIR"
echo "Claude: $CLAUDE_DIR"
echo "Agent:  $RUNPOD_AGENT_HOME"
echo

# 1. Claude Code 에이전트
echo "[1/3] Claude Code agent 설치"
cp -v "$REPO_DIR/agents/runpod-deploy.md" "$CLAUDE_AGENTS/runpod-deploy.md"

# 2. Hook 스크립트
echo
echo "[2/3] Hook 스크립트 설치"
for hook in deploy-preflight-guard.py runpod-guard.js; do
  if [ -f "$REPO_DIR/hooks/$hook" ]; then
    cp -v "$REPO_DIR/hooks/$hook" "$CLAUDE_HOOKS/$hook"
    chmod +x "$CLAUDE_HOOKS/$hook"
  fi
done

# 3. 파이프라인 도구
echo
echo "[3/3] 파이프라인 도구 설치"
cp -rv "$REPO_DIR/tools" "$RUNPOD_AGENT_HOME/"
cp -rv "$REPO_DIR/rules" "$RUNPOD_AGENT_HOME/"
cp -rv "$REPO_DIR/examples" "$RUNPOD_AGENT_HOME/"

# PATH 편의 심볼릭 링크
mkdir -p "$HOME/.local/bin"
ln -sf "$RUNPOD_AGENT_HOME/tools/preflight-guard.py" "$HOME/.local/bin/runpod-preflight"
ln -sf "$RUNPOD_AGENT_HOME/tools/pod-deploy-guard.py" "$HOME/.local/bin/runpod-deploy"
ln -sf "$RUNPOD_AGENT_HOME/tools/idle-monitor-guard.py" "$HOME/.local/bin/runpod-idle-monitor"

echo
echo "=== 설치 완료 ==="
echo
echo "다음 단계:"
echo
echo "1. Claude Code settings.json 에 hook 등록:"
echo
cat <<'HOOK_JSON'
   {
     "hooks": {
       "PreToolUse": [
         {
           "matcher": "Write|Edit",
           "hooks": [
             {"type": "command", "command": "python3 ~/.claude/hooks/scripts/deploy-preflight-guard.py"}
           ]
         },
         {
           "matcher": "Bash",
           "hooks": [
             {"type": "command", "command": "node ~/.claude/hooks/scripts/runpod-guard.js"}
           ]
         }
       ]
     }
   }
HOOK_JSON
echo
echo "2. RunPod API 키 설정:"
echo "   echo YOUR_RUNPOD_API_KEY > ~/.runpod_api_key"
echo "   chmod 600 ~/.runpod_api_key"
echo "   export RUNPOD_API_KEY=\$(cat ~/.runpod_api_key)"
echo
echo "3. Claude Code에서 에이전트 호출:"
echo "   /agent runpod-deploy"
echo
echo "4. CLI 도구:"
echo "   runpod-preflight your_script.py"
echo "   runpod-deploy --list"
echo "   runpod-idle-monitor --action report"
echo
echo "참고 문서:"
echo "  - $RUNPOD_AGENT_HOME/rules/03-core-rules.md"
echo "  - $RUNPOD_AGENT_HOME/rules/04-bnb-4bit-recipe.md"
echo "  - $RUNPOD_AGENT_HOME/rules/01-crash-catalog.md"
