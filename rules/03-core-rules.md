# Core Rules (R0~R11)

각 규칙은 실제 사건에서 도출됨. 위반 시 비용/시간 손실 큼.

---

## R0. 독자적 Pod 생성/재생성 절대 금지 (CRITICAL)

- Pod 생성은 명시적 승인 후에만 허용
- Agent 자체 판단으로 Pod 생성/재시도 금지
- 실패 시: 보고 → 승인 → 재생성

**근거**: 자동 재시도 루프가 크래시를 방치한 채 credit만 소진시킴.

---

## R1. Pod idle 방치 절대 금지

- N개 jobs → `ceil(N/vCPU)` Pods
- 가용 Pod 전체에 균등 분배
- Pod 간 job 수 불균형 금지

**재분배 공식**:
```python
jobs_per_pod = math.ceil(len(remaining_jobs) / len(available_pods))
```

**근거**: 1-job pod이 8-job pod 완료 대기 중 idle로 과금.

---

## R2. Pod 생성 후 SSH 검증 필수

- SSH 접속 테스트: 3회 retry, 10초 간격
- 실패 → terminate → 새 Pod
- 2회 연속 실패 → 5분 대기 → 재시도
- `dmesg | grep -i error` — hardware alert 발견 시 즉시 교체

**근거**: Pod 생성은 성공해도 SSH가 막힌 pod 존재 (이미지 pull 미완료, 호스트 이슈 등).

---

## R3. 로그는 terminate 전에 반드시 회수

- cleanup 순서: **`fetch_logs` → `terminate`** (역순 절대 금지)
- `except`, `finally` 모두에서 `fetch_logs` 호출
- 모니터링 중 중간 rsync 실행 (2분 간격)
- SSH 불가 시: `stop` (terminate 아님) → SSH 재시도 → 로그 회수

### R3-A. Pod별 개별 완료 즉시 처리 (CRITICAL)

- **모든 pod 대기 금지**. 완료 즉시 처리.
- 모니터링 루프 내:
  1. 해당 Pod 로그만 rsync
  2. 로컬 파일 수 검증 (기대값 일치)
  3. 즉시 terminate
- 폴링 간격: `sleep 120` (2분). `sleep 600`+ 사용 절대 금지

**근거**: 완료된 pod idle 방치 = 다른 pod 대기 중 과금 누적.

---

## R4. MAX_PARALLEL = vCPU 수

- RTX 3090 Spot: vCPU 4 → MAX_PARALLEL=4
- A100 SXM: vCPU 8~16 → MAX_PARALLEL=8
- 동시 프로세스 > vCPU → CPU 100% → SSH 불능
- 다수 pod 생성 시 10초 간격 (API rate limit)

---

## R5. 실험 공통 기반(COMBO) 검증

- 모든 arm이 공유하는 플래그/config 먼저 확정
- launcher 완성 후 각 arm 최종 명령줄 출력하여 리뷰

**근거**: COMBO에 핵심 플래그 하나 누락되면 전체 실험 무효.

---

## R6. queue_runner.py 패턴 필수

- Pod에 jobs.txt + queue_runner.py 업로드
- `nohup python -u queue_runner.py {MAX_PARALLEL} > logs/queue_runner.log 2>&1 &`
- poll: `tail -1 queue_runner.log` + `ps aux | grep run_study | wc -l`
- **직접 `nohup`로 job launch 금지** — SSH 세션 종료 시 SIGHUP로 프로세스 사망

---

## R7. 재배포 시 로컬 완료 로그 확인 필수

- 재배포 전 `ls logs/{name}/study_reports/ | wc -l` → 완료 수 확인
- **전체 재배포 절대 금지** — 누락분만 선별 배포
- "완료 N개, 누락 M개, M개만 배포" 보고 후 진행

---

## R8. 세션 전환 시 기존 Pod 처리

기존 Pod 발견 시: terminate 말고 상태 확인부터.

1. SSH → queue_runner 진행 상태 확인
2. 실행 중 → 대기 또는 중간 rsync
3. 완료 → rsync → terminate
4. SSH 불가 → `stop` → 재시도 → 로그 회수

**절대 금지**: "일단 terminate하고 새로 시작"

---

## R9. Launch 후 60초 Double-Check 필수

- 모든 Pod에 jobs 배포 후 60초 대기 → `ps aux | grep YOUR_SCRIPT | wc -l`
- procs=0 발견 시:
  1. queue_runner 생존 확인
  2. dead → 재업로드 → 재실행
  3. 60초 후 2차 확인 → 여전히 0 → failed 처리
- **"launch 했으니 돌아가겠지" 가정 금지. 실측 확인.**

---

## R10. 중복 Pod 생성 금지

- Pod 생성 전 **반드시** 기존 pod 조회
- idle Pod → 재활용 (코드 재배포 + runner 재시작)
- failed Pod → 로그 회수 → terminate → 그 후에만 새 Pod 생성
- `wait_for_pod` 기본값: 1200초 (SSH 600초 timeout 사고 재발 방지)

---

## R11. Seed 병렬 실행 필수 (GPU 활용도)

- `--n-seeds N` 내부 순차 루프 **금지** — GPU 유휴 발생
- **올바른 방법**: `--seed {i} --n-seeds 1` × N개를 `Popen`으로 동시 launch

```python
procs = []
for i, seed in enumerate([42, 43, 44]):
    log = open(f"logs/seed{seed}.log", "w")
    p = Popen([..., "--seed", str(seed), "--n-seeds", "1",
               "--out", f"result_s{seed}.json"], stdout=log, stderr=log)
    procs.append(p)
for p in procs: p.wait()
```

**예외**: 단일 프로세스가 GPU 100% 점유 시 순차 실행 허용 (VRAM 부족).

---

## R12. 대형 모델 로딩은 정적 분석 먼저

3시간+ 걸리는 로딩 작업은 시작 전에 반드시:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정 확인
2. `max_memory`에 `"cpu"` 엔트리 없음 확인
3. BnB 4-bit 사용 시 **수동 device_map** 작성 확인
4. Forward hook 함수 내부에서 텐서 CPU 연산 확인
5. 사용 라이브러리 버전 + 모델 조합의 known issue 검색

**근거**: 크래시 → 수정 → 재시도 반복은 $30~$100 단위 손실. 정적 분석으로 사전 차단.

---

## R13. 장기 작업 on-demand 강제

1시간 이상 걸리는 작업은 **반드시 `interruptible: false`** (on-demand).

- on-demand $X/hr vs spot $Y/hr 차이는 보통 15~20%
- spot 뺏기면 그 시간까지의 작업 전부 소실
- 기대값: `cost_ondemand = X × T`, `cost_spot_risked = Y × T × (1 + p_interrupted × 1.5)`
- p_interrupted > 20%면 on-demand가 더 싸다

**예외**: < 15분 작업 + resume 가능한 경우만 spot 허용.

---

## R14. Network Volume 삭제 금지

- `deleteNetworkVolume` GraphQL mutation 호출 금지
- MCP `delete-network-volume` 도구 호출 금지
- 명시적 삭제 승인이 있을 때만 허용
- DC 변경 필요 시: 새 볼륨 생성 → 데이터 복사 → 승인 → 구 볼륨 삭제

**근거**: 수 GB~TB 모델 데이터 소멸 시 수시간 재다운로드 비용.
