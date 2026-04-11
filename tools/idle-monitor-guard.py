#!/usr/bin/env python3
"""
Idle Monitor Guard — 유휴 RunPod 자동 감지 및 stop

일정 시간 이상 GPU 0% 상태인 pod을 자동으로 stop.

Usage:
    export RUNPOD_API_KEY=$(cat ~/.runpod_api_key)
    python idle-monitor-guard.py \
        --idle-threshold-min 30 \
        --poll-interval-sec 900 \
        --action stop
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("[ERROR] RUNPOD_API_KEY 미설정", file=sys.stderr)
        sys.exit(2)
    return key


def _graphql(query: str) -> dict:
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        RUNPOD_GRAPHQL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
            "User-Agent": "curl/7.88.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode()).get("data", {})


def list_running_pods() -> list:
    query = """
    {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime {
            uptimeInSeconds
            ports {
              ip
              isIpPublic
              privatePort
              publicPort
            }
          }
        }
      }
    }
    """
    data = _graphql(query)
    pods = (data.get("myself") or {}).get("pods") or []
    return [p for p in pods if p.get("desiredStatus") == "RUNNING"]


def ssh_nvidia_smi(ip: str, port: int) -> float | None:
    """SSH로 nvidia-smi 실행해서 GPU utilization 평균 반환 (0-100)."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-p", str(port),
        f"root@{ip}",
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return None
        utilizations = [
            int(line.strip())
            for line in result.stdout.strip().split("\n")
            if line.strip().isdigit()
        ]
        if not utilizations:
            return None
        return sum(utilizations) / len(utilizations)
    except Exception as e:
        print(f"[WARN] SSH failed for {ip}:{port}: {e}", file=sys.stderr)
        return None


def stop_pod(pod_id: str) -> bool:
    mutation = """
    mutation { podStop(input: {podId: "%s"}) { id } }
    """ % pod_id
    try:
        _graphql(mutation)
        return True
    except Exception as e:
        print(f"[ERROR] stop failed for {pod_id}: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idle-threshold-min", type=int, default=30, help="유휴 판정 시간(분)")
    p.add_argument("--poll-interval-sec", type=int, default=900, help="폴링 주기(초)")
    p.add_argument("--action", choices=["stop", "report"], default="report")
    p.add_argument("--once", action="store_true", help="한 번만 실행 후 종료")
    args = p.parse_args()

    idle_history: dict[str, int] = {}  # pod_id → consecutive idle polls

    while True:
        pods = list_running_pods()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Running pods: {len(pods)}")

        current_ids = set()
        for pod in pods:
            pod_id = pod["id"]
            current_ids.add(pod_id)
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            ssh_port = next(
                (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
                None,
            )
            if not ssh_port:
                print(f"  {pod_id} ({pod.get('name')}): no public SSH → SKIP")
                continue

            util = ssh_nvidia_smi(ssh_port["ip"], ssh_port["publicPort"])
            if util is None:
                print(f"  {pod_id} ({pod.get('name')}): SSH unreachable → SKIP")
                continue

            is_idle = util < 1.0
            history_count = idle_history.get(pod_id, 0)
            if is_idle:
                idle_history[pod_id] = history_count + 1
            else:
                idle_history[pod_id] = 0

            polls_needed = max(
                1, (args.idle_threshold_min * 60) // args.poll_interval_sec
            )
            status = "IDLE" if is_idle else "ACTIVE"
            print(
                f"  {pod_id} ({pod.get('name')}): GPU={util:.1f}% {status} "
                f"({idle_history[pod_id]}/{polls_needed} polls)"
            )

            if idle_history[pod_id] >= polls_needed:
                if args.action == "stop":
                    print(f"    → IDLE 임계 초과. Stop 실행.")
                    if stop_pod(pod_id):
                        print(f"    → Stopped: {pod_id}")
                        del idle_history[pod_id]
                else:
                    print(f"    → IDLE 임계 초과 (report only).")

        # 종료된 pod의 history 정리
        for gone_id in list(idle_history.keys()):
            if gone_id not in current_ids:
                del idle_history[gone_id]

        if args.once:
            break
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    main()
