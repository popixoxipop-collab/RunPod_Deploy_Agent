# RunPod API / GraphQL 특이사항

실전에서 부딪힌 API 함정들.

---

## 인증 헤더

**잘못된 예**:
```bash
curl -H "api-key: $API_KEY" https://api.runpod.io/graphql ...
# 결과: myself: null (인증 실패)
```

**올바른 예**:
```bash
curl -H "Authorization: Bearer $API_KEY" https://api.runpod.io/graphql ...
```

**Python urllib 주의**: 기본 User-Agent는 403 차단됨. `curl/7.88.1` 등으로 override 필수.

---

## `podFindAndDeployOnDemand` 필수 필드

```graphql
mutation {
  podFindAndDeployOnDemand(input: {
    cloudType: ALL,           # SECURE만으론 재고 부족 빈발
    gpuCount: 1,              # 필수! 누락 시 SUPPLY_CONSTRAINT
    gpuTypeId: "NVIDIA A100-SXM4-80GB",
    volumeInGb: 0,            # Network volume 사용 시 0
    containerDiskInGb: 20,
    dataCenterId: "US-MD-1",  # Network volume과 같은 DC 필수
    networkVolumeId: "YOUR_VOL_ID",
    imageName: "runpod/pytorch:...",
    env: [
      {key: "PUBLIC_KEY", value: "ssh-ed25519 AAAA... user"},
      {key: "HF_TOKEN", value: "..."},
    ],
    ports: "22/tcp",
    startSsh: true,
    supportPublicIp: true,
    # interruptible 필드 생략하면 on-demand, true면 spot
  }) {
    id
    machineId
    imageName
  }
}
```

### 삭제된 필드 (사용 금지)

- `gpuDisplayName` — GraphQL validation error 유발

---

## Spot/On-Demand 구분

- **On-demand**: `interruptible` 필드 생략 또는 `false`
- **Spot**: `interruptible: true` + `bidPerGpu: X.XX` 필수
  - bidPerGpu 누락 시 무효
  - 너무 낮으면 할당 실패
  - 실제 요금은 spot 시장가 (최대 bid 이하)

---

## Network Volume 제약

1. **DC 고정**: 볼륨은 한 DC에만 존재. Pod는 같은 DC여야 마운트 가능
   ```
   Volume DC = US-MD-1 → Pod도 dataCenterId: "US-MD-1" 필수
   ```
2. **다른 DC Pod 생성 시**: 볼륨 마운트 실패 → pod 무용지물
3. **볼륨 삭제는 불가역**: `deleteNetworkVolume` 호출 전 명시적 승인 필수

---

## `publicIp`, `portMappings` 필드 부재

**MCP 서버**에서는 `get-pod` 결과에 `publicIp`, `portMappings` 포함됨.
**Raw GraphQL**에서는 **없음**. 대신:
```graphql
{
  pod(input: {podId: "XXX"}) {
    runtime {
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
```
에서 `ip` + `publicPort` 조합으로 SSH 주소 구성.

---

## SSH 접속

### SSH 설정 요구

- Pod env에 `PUBLIC_KEY` 포함 (SSH 공개키 직접)
- 또는 RunPod 계정 SSH 키 등록 (모든 pod에 자동 적용)

### SSH 커맨드

```bash
ssh -o StrictHostKeyChecking=no \
    -o ConnectTimeout=30 \
    -p ${PORT} \
    root@${IP}
```

### 주의

- `stdin`은 `/dev/null`로 리다이렉트 (SSH hang 방지)
  ```python
  subprocess.run(["ssh", ...], stdin=subprocess.DEVNULL)
  ```
- 장시간 명령은 background + 로그 redirect
  ```bash
  ssh ... "nohup python script.py > log 2>&1 </dev/null &"
  ```

---

## 이미지 pull 지연

- 대형 이미지(9+ GB) + 미캐시 머신 → 수십 분 소요
- 대응:
  1. 이미 캐시된 머신 재사용 (이전 성공 pod의 machineId)
  2. `cloudType: SECURE` 우선 (캐시 확률 높음)
  3. 경량 이미지 대체: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

---

## Rate Limit

- Pod 생성 연속 호출 간 **10초 간격** 권장
- 다수 pod 배포 시 for 루프에 `time.sleep(10)` 삽입

---

## `wait_for_pod` 타임아웃

- 기본값 1200초 (20분) 권장
- 300초는 이미지 pull 시간 고려 안 함 → 타임아웃 사고
- 대형 이미지는 `2400` (40분)도 고려

---

## MCP 서버 사용 (Claude Code)

```json
// ~/.claude.json
"mcpServers": {
  "runpod": {
    "command": "npx",
    "args": ["-y", "@runpod/mcp-server"],
    "env": {"RUNPOD_API_KEY": "..."}
  }
}
```

사용 가능한 tools:
- `mcp__runpod__list-pods`
- `mcp__runpod__get-pod`
- `mcp__runpod__create-pod`
- `mcp__runpod__stop-pod`
- `mcp__runpod__delete-pod`
- `mcp__runpod__list-network-volumes`
- `mcp__runpod__create-network-volume`
- `mcp__runpod__delete-network-volume` (사용 금지)
- `mcp__runpod__start-pod`

GraphQL 대비 장점: 인증 자동, 결과 포맷 정리됨, typed schema.
