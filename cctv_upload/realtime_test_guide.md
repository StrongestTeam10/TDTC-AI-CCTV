# 📡 실시간 스트리밍 파이프라인 테스트 가이드

## 1️⃣ 준비 단계

| 항목 | 설명 | 명령·설정 |
|------|------|-----------|
| **가상 환경** | 의존성 격리 | `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt` |
| **CUDA & GPU** | RTX 4060 사용 여부 확인 | `nvidia-smi` (CUDA 12 이상 권장) |
| **테스트용 RTSP 스트림** | 실제 카메라가 없을 때 가상 스트림 사용 | ```bash
# sample.mp4 를 무한 루프 RTSP 로 스트리밍
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/stream1
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/stream2
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/stream3
``` |
| **config.yaml** (실시간 섹션) | 스트림 URL, FPS 목표, 비디오 저장 옵션 지정 | ```yaml
pipeline_mode: realtime
realtime:
  streams:
    - name: zone1
      url: rtsp://127.0.0.1:8554/stream1
    - name: zone2
      url: rtsp://127.0.0.1:8554/stream2
    - name: zone3
      url: rtsp://127.0.0.1:8554/stream3
  fps_target: 10          # 목표 프레임/초 (각 구역당)
  save_video: false       # true 로 하면 결과 비디오 저장 (optional)
``` |

> **Tip**: `fps_target` 를 실제 스트림 FPS와 맞추면 프레임 드롭을 최소화할 수 있습니다.

---

## 2️⃣ 실행 명령

```bash
# 실시간 모드 시작
python run_pipeline.py --mode realtime
```

- `run_pipeline.py` → `realtime_stream.py` 를 호출합니다.
- 내부 흐름 (요약):
```
read_rtsp_stream → grab_frame (3 streams) → batch(3 frames) → run_inference → postprocess → async_insert → logger → repeat
```

---

## 3️⃣ 검증 포인트 (콘솔/로그)

| 체크 포인트 | 로그 키 | 기대값 |
|-------------|--------|--------|
| **스트림 연결** | `STREAM_CONNECTED` | 3개의 `rtsp://...` 가 모두 `Connected` 로 표시 |
| **프레임 캡처** | `frame_timestamp` / `fps` | 목표 FPS(≈10) 근처, `pipeline_log.jsonl` 에 매 초마다 3개의 프레임 로그 |
| **추론 지연** | `inference_time_ms` | 20 ~ 30 ms 이하 (GPU 가속 기준) |
| **DB 삽입** | `async_insert_success` | 각 프레임마다 `True` 가 기록, Supabase 테이블에 실시간으로 레코드 증가 |
| **전체 레이턴시** | `end_to_end_latency_ms` (프레임 → DB 삽입까지) | 30 ~ 50 ms 이하 (실시간 대시보드에 충분) |
| **비디오 저장** (옵션) | 파일 존재 여부 | `results/realtime_output.mp4` 가 생성 (옵션 켜면) |

> 로그는 `cctv_upload/pipeline_log.jsonl` 에 JSON 라인 형태로 기록됩니다. `grep "async_insert_success"` 로 삽입 성공 여부를 빠르게 확인할 수 있습니다.

---

## 4️⃣ 자동 테스트 스크립트 (pytest 예시)

```python
import asyncio, json, os, pathlib
from cctv_upload.realtime_stream import run_realtime

async def _run_for(seconds: int = 20):
    # 20초 동안 실행하고 자동 종료
    await asyncio.wait_for(run_realtime(max_seconds=seconds), timeout=seconds+5)

def test_realtime_basic(tmp_path: pathlib.Path):
    # 로그 위치 지정 (환경변수 사용)
    log_path = tmp_path / "realtime_log.jsonl"
    os.environ["REALTIME_LOG"] = str(log_path)

    asyncio.run(_run_for(15))

    # 로그 파싱
    lines = log_path.read_text().splitlines()
    assert any("STREAM_CONNECTED" in l for l in lines), "스트림 연결 로그 없음"
    assert any("async_insert_success" in l for l in lines), "DB 삽입 로그 없음"
    # 최소 5개의 프레임이 캡처됐는지 확인
    frame_cnt = sum(1 for l in lines if '"type":"frame"' in l)
    assert frame_cnt >= 5, f"프레임 수가 부족함: {frame_cnt}"
```

- **CI** 에서 위 테스트를 실행하면 가상 RTSP 스트림을 이용해 실시간 파이프라인이 정상 동작함을 자동 검증할 수 있습니다.

---

## 5️⃣ 디버깅 팁

| 문제 | 원인 | 해결 방법 |
|------|------|----------|
| **스트림 연결 실패** | URL 오타, 포트 차단 | `ffmpeg -rtsp_transport tcp -i rtsp://...` 로 직접 재생해 확인 |
| **프레임 드롭** | `fps_target` > 실제 스트림 FPS | `ffprobe -v error -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 rtsp://...` 로 FPS 확인 후 `config.yaml` 조정 |
| **GPU OOM** | 배치 크기 과다 | `realtime_stream.py` 에서 `batch_size = 1` 로 고정 (실시간은 기본 1) |
| **DB Rate‑Limit** | 초당 INSERT 횟수 초과 | `async_insert` 에 `await asyncio.sleep(0.01)` 삽입 혹은 **배치 삽입**(예: 10프레임마다 `bulk_insert`) 로 전환 |
| **로그 파일이 비어 있음** | `REALTIME_LOG` 환경변수 누락 | `set REALTIME_LOG=./pipeline_log.jsonl` (Windows) 혹은 `export REALTIME_LOG=./pipeline_log.jsonl` |

---

## 6️⃣ 전체 흐름 요약

1️⃣ **환경** → 가상 env + CUDA + FFmpeg 로 가상 RTSP 스트림 준비
2️⃣ **설정** → `config.yaml` 에 스트림 URL·FPS·저장 옵션 입력
3️⃣ **실행** → `python run_pipeline.py --mode realtime`
4️⃣ **검증** → 콘솔 로그 + `pipeline_log.jsonl` + Supabase 테이블 확인
5️⃣ **옵션** → `save_video: true` 로 결과 비디오 저장 가능

위 과정을 그대로 따라 하면 **실시간 파이프라인을 로컬에서 손쉽게 테스트**할 수 있습니다. 추가적인 테스트 시나리오(예: 멀티 GPU, TensorRT 적용) 혹은 특정 오류 상황에 대한 진단이 필요하면 언제든 알려 주세요! 🚀

## 7️⃣ 실시간 vs 배치 모드 통합 설계

- **자동 감지** (`--mode auto`) 를 기본값으로 두어 `config.yaml` 에 `realtime.streams` 가 있으면 실시간, 없으면 배치가 자동 선택됩니다.
- **명시적 전환**: 필요 시 `--mode realtime` 혹은 `--mode batch` 로 강제 전환 가능합니다.
- **서버 배포**: 하나의 Docker 이미지·엔트리 포인트(`run_pipeline.py`) 로 두 모드 모두 지원해 인프라 관리가 간편해집니다.
- **대시보드 연동**: 실시간 스트림에서는 `WebSocket` 으로 바로 프레임 데이터를 전송하고, 배치 작업은 완료 후 일괄 DB 적재 후 대시보드가 조회하도록 설계합니다.

이렇게 하면 **실시간이 기본 흐름**이면서도 **배치 작업을 필요 시 별도 호출**할 수 있어, 불필요한 모드 전환 없이도 요구사항을 모두 충족할 수 있습니다.
