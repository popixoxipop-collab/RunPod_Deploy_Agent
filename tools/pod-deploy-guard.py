#!/usr/bin/env python3
"""
Pod Deployment Guard — 안전한 RunPod Pod 생성 CLI

실전 시행착오에서 도출된 규칙을 강제하는 배포 도구:
- 기존 pod 조회 (중복 생성 방지, R10)
- Spot instance 차단 (장기 작업, R13)
- PUBLIC_KEY, networkVolumeId 누락 차단
- `gpuCount` 필드 필수 (SUPPLY_CONSTRAINT 방지)
- `Authorization: Bearer` 헤더 강제
- DC-볼륨 일치 확인

Usage:
    export RUNPOD_API_KEY=$(cat ~/.runpod_api_key)
    python pod-deploy-guard.py --list
    python pod-deploy-guard.py \
        --gpu-type "NVIDIA A100-SXM4-80GB" \
        --gpu-count 6 \
        --volume-id vol_xxx \
        --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
        --public-key "$(cat ~/.ssh/id_ed25519.pub)" \
        --datacenter US-MD-1
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("[ERROR] RUNPOD_API_KEY 환경변수 미설정", file=sys.stderr)
        print("해결: export RUNPOD_API_KEY=$(cat ~/.runpod_api_key)", file=sys.stderr)
        sys.exit(2)
    return key


def _graphql(query: str, variables: dict | None = None) -> dict:
    """RunPod GraphQL 호출 (Bearer 헤더 강제)."""
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        RUNPOD_GRAPHQL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
            "User-Agent": "curl/7.88.1",  # urllib 기본 UA는 403 차단됨
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            result = json.loads(body)
            if "errors" in result:
                print(f"[GraphQL errors] {result['errors']}", file=sys.stderr)
                sys.exit(1)
            return result.get("data", {})
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}] {e.read().decode()}", file=sys.stderr)
        sys.exit(1)


def list_pods() -> list:
    """기존 pod 목록 조회."""
    query = """
    {
      myself {
        pods {
          id
          name
          desiredStatus
          gpuCount
          costPerHr
          machineId
          runtime {
            uptimeInSeconds
            ports {
              ip
              isIpPublic
              privatePort
              publicPort
              type
            }
          }
        }
      }
    }
    """
    data = _graphql(query)
    return (data.get("myself") or {}).get("pods") or []


def list_volumes() -> list:
    query = """
    {
      myself {
        networkVolumes {
          id
          name
          size
          dataCenterId
        }
      }
    }
    """
    data = _graphql(query)
    return (data.get("myself") or {}).get("networkVolumes") or []


def create_pod(args) -> str:
    """Pod 생성 — 모든 규칙 강제."""
    # 1. 볼륨 DC 확인 (R: volume-DC mismatch)
    volumes = list_volumes()
    vol_info = next((v for v in volumes if v["id"] == args.volume_id), None)
    if not vol_info:
        print(f"[FAIL] Volume {args.volume_id} 없음", file=sys.stderr)
        sys.exit(1)
    if vol_info["dataCenterId"] != args.datacenter:
        print(
            f"[FAIL] Volume DC={vol_info['dataCenterId']} ≠ Pod DC={args.datacenter}. "
            f"볼륨과 pod는 같은 DC에 있어야 마운트 가능.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. 기존 pod 조회 (R10)
    existing = list_pods()
    running = [p for p in existing if p.get("desiredStatus") == "RUNNING"]
    if running and not args.force:
        print(f"[WARN] 이미 {len(running)}개 RUNNING pod 존재:")
        for p in running:
            print(f"  - {p['id']} ({p.get('name', '?')}) {p.get('gpuCount')} GPU ${p.get('costPerHr')}/hr")
        print("계속하려면 --force 플래그 사용")
        sys.exit(0)

    # 3. PUBLIC_KEY 필수 (SSH 접속용)
    if not args.public_key:
        print("[FAIL] --public-key 필수 (SSH 공개키)", file=sys.stderr)
        sys.exit(1)

    # 4. Spot instance 차단 (R13)
    if args.spot:
        if not args.allow_spot:
            print(
                "[FAIL] --spot 지정했지만 --allow-spot 없음. "
                "장기 작업에 spot 사용 금지 (R13). "
                "짧은 작업(<15분)만 --allow-spot 허용.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 5. GraphQL mutation
    env_payload = [
        {"key": "PUBLIC_KEY", "value": args.public_key},
    ]
    if args.hf_token:
        env_payload.append({"key": "HF_TOKEN", "value": args.hf_token})

    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        machineId
        imageName
      }
    }
    """
    input_obj = {
        "cloudType": args.cloud_type,
        "gpuCount": args.gpu_count,  # 필수 (SUPPLY_CONSTRAINT 방지)
        "gpuTypeId": args.gpu_type,
        "volumeInGb": 0,
        "containerDiskInGb": args.container_disk_gb,
        "dataCenterId": args.datacenter,
        "networkVolumeId": args.volume_id,
        "imageName": args.image,
        "env": env_payload,
        "ports": "22/tcp",
        "startSsh": True,
        "supportPublicIp": True,
    }
    if args.spot and args.allow_spot:
        input_obj["interruptible"] = True
        input_obj["bidPerGpu"] = args.bid_per_gpu

    data = _graphql(mutation, {"input": input_obj})
    pod = data.get("podFindAndDeployOnDemand")
    if not pod:
        print("[FAIL] Pod 생성 실패", file=sys.stderr)
        sys.exit(1)

    pod_id = pod["id"]
    print(f"[OK] Pod 생성됨: {pod_id}")
    print(f"     machine: {pod['machineId']}")
    print(f"     image:   {pod['imageName']}")
    print(f"\nSSH 대기 중 (최대 {args.ssh_wait_seconds}초)...")
    return pod_id


def main():
    p = argparse.ArgumentParser(description="RunPod Pod Deployment Guard")
    p.add_argument("--list", action="store_true", help="기존 pod 목록 조회")
    p.add_argument("--list-volumes", action="store_true", help="Network volume 목록 조회")

    p.add_argument("--gpu-type", default="NVIDIA A100-SXM4-80GB")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--volume-id", help="Network volume ID")
    p.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    p.add_argument("--datacenter", default="US-MD-1")
    p.add_argument("--cloud-type", default="ALL", choices=["ALL", "SECURE", "COMMUNITY"])
    p.add_argument("--container-disk-gb", type=int, default=20)
    p.add_argument("--public-key", help="SSH 공개키 (파일 경로 아님)")
    p.add_argument("--hf-token", help="HuggingFace 토큰")

    p.add_argument("--spot", action="store_true", help="Spot(interruptible) 인스턴스")
    p.add_argument(
        "--allow-spot",
        action="store_true",
        help="15분 미만 작업에만 허용 (R13)",
    )
    p.add_argument("--bid-per-gpu", type=float, default=0.22)

    p.add_argument("--force", action="store_true", help="기존 pod 있어도 강제 생성")
    p.add_argument("--ssh-wait-seconds", type=int, default=1200)

    args = p.parse_args()

    if args.list:
        pods = list_pods()
        if not pods:
            print("(pod 없음)")
        for pod in pods:
            runtime = pod.get("runtime") or {}
            uptime = runtime.get("uptimeInSeconds") or 0
            ports = runtime.get("ports") or []
            ssh_port = next(
                (p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")),
                None,
            )
            ssh_str = (
                f"{ssh_port['ip']}:{ssh_port['publicPort']}"
                if ssh_port else "(no public ssh)"
            )
            print(
                f"{pod['id']}  {pod.get('name', '?'):30s}  "
                f"{pod.get('desiredStatus', '?'):10s}  "
                f"{pod.get('gpuCount', 0)} GPU  "
                f"${pod.get('costPerHr', 0):.2f}/hr  "
                f"uptime={uptime // 60}m  ssh={ssh_str}"
            )
        return

    if args.list_volumes:
        vols = list_volumes()
        for v in vols:
            print(f"{v['id']}  {v['name']:30s}  {v['size']} GB  {v['dataCenterId']}")
        return

    if not args.volume_id:
        print("[FAIL] --volume-id 필수", file=sys.stderr)
        sys.exit(2)

    create_pod(args)


if __name__ == "__main__":
    main()
