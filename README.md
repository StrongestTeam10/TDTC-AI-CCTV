# 🎥 TDTC CCTV AI 통합 분석 서버 2.2 (TDTC-AI-CCTV)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![YOLO11](https://img.shields.io/badge/YOLO-11n-00FFFF?style=for-the-badge&logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Cloudflare Tunnel](https://img.shields.io/badge/Cloudflare-Permanent%20Domain-F38020?style=for-the-badge&logo=cloudflare)](https://tdtc-ai-cctv.uk)
[![Swagger Docs](https://img.shields.io/badge/Swagger%20UI-Live%20Docs-0055FF?style=for-the-badge&logo=swagger)](https://tdtc-ai-cctv.uk/docs)

망원시장 스마트 CCTV의 다중 구역(Zone 1 남측, Zone 2 중앙, Zone 3 북측)을 관제하는 **실시간 10 FPS AI 영상 스트리밍(모자이크 비식별화), 3개 구역 실시간 동기화 관제, 정체시간 산출, PostgreSQL RDS 적재, 긴급 자동/수동 알람 웹훅 중계 서버**입니다.

---

## 🌟 핵심 아키텍처 및 기능 (Core Features)

### 1. 🤖 CSRNet + YOLO11n 앙상블 군중 분석 & 정체 시간 산출
* **YOLO11n + CSRNet**: 초고속 인물 바운딩 박스 검출 및 군중 밀도 지도(Density Map) 앙상블 추론
* **가우시안 모자이크(비식별화)**: 개인정보 보호를 위한 보행자 영역 실시간 블러 처리 (초당 10 FPS 스트리밍)
* **BEV (Bird's Eye View) 물리 좌표 변환**: Homography ($H$-Matrix) 기반으로 CCTV 2D 픽셀을 실세계 미터 단위 평면 좌표계로 역투영
* **정체 시간(`stagnation_sec`) & CRIv3 안정화**: 인파 체류 시간을 EMA 모델로 누적/해소하고, 이중 EMA 스무딩을 적용하여 튀는 현상 없는 안정적 위험도 지표 제공

### 2. 📡 실시간 3개 구역 10 FPS 스트리밍 & 동기화 WebSocket
* **WebSocket 엔드포인트**: `wss://tdtc-ai-cctv.uk/ws/cctv-stream` (또는 `ws://localhost:8088/ws/cctv-stream`)
* **3개 구역 전역 동기화**: Zone 1, Zone 2, Zone 3의 최신 인원수/위험도 지표를 동일한 동기화 시퀀스로 전송하여 대시보드 전체 합계(Total)가 튀지 않고 정확하게 합산
* **10 FPS 경량 스트리밍**: 1.0x 정상 배속을 유지하면서 연산 부하 및 네트워크 대역폭 67% 대폭 절감

### 3. 🌐 Cloudflare 영구 고정 도메인 인프라
* **영구 고정 도메인**: **`https://tdtc-ai-cctv.uk`**
* **무경고/무제한 대역폭**: 브라우저 경고창(Visit Site) 없이 즉시 영상 송출 및 대역폭 제한 없는 24시간 안정적 터널링 가동

### 4. 🗄️ PostgreSQL RDS 데이터베이스 자동 적재
* **psycopg2 커넥터**: 비디오 분석 완료 시 추출된 보행자 물리 좌표(`pedcord01h`) 및 혼잡도 이력을 원격 RDS(PostgreSQL 17.6)에 자동 적재

### 5. 🚨 실시간 자동 위험 감지 및 3분 쿨다운 엔진 (Auto-Alert)
* **위험 감지 기준**: CRI 점수 70pt 이상 위험 상황이 **10초간 연속 지속**될 때 AI 서버가 스스로 판단
* **비동기 자동 발령**: 35초 위험 클립 추출 + 사고 명세서 PDF 생성 + S3 업로드 + Java 백엔드 `AUTO_REPORT` 웹훅 전송
* **쿨다운 타이머**: 동일 구역 내 3분(180초) 동안 중복 알람 방지

### 6. 📼 35초 위험 클립 버퍼링 & 1분 상시 영상 분할
* **Rolling Frame Buffer**: 메모리 상의 원형 큐로 최근 35~40초간의 프레임을 상시 보관하여 출동 시 즉시 비상 클립 추출 (`clipType: "RISK"`, S3 `danger-clips/` 및 `post-reports/`)
* **Raw Video Splitter**: 1분 단위 상시 영상 분할 인코딩 (`clipType: "TEMP"`, S3 `raw-videos/`)

---

## 🛠️ 핵심 성능 최적화 및 트러블슈팅 (Optimization)

| 해결 과제 | 기존 문제점 | 적용 해결 기법 | 최종 개선 성과 |
| :--- | :--- | :--- | :--- |
| **대역폭 한도 및 경고창** | 무료 터널 대역폭 1GB 초과 및 Visit 경고창 발생 | **Cloudflare 영구 고정 도메인(`tdtc-ai-cctv.uk`)** | 24시간 무제한 스트리밍 & 경고창 0% |
| **대시보드 전체 합계 튐** | 3개 구역 프레임 ID 불일치로 0.1초마다 합계 요동침 | **전역 상태 보존 및 동기화 프레임 브로드캐스트** | 3개 구역 합계가 튀지 않고 온전하게 실시간 표출 |
| **스트리밍 부하 과다** | 30 FPS 전송으로 인한 GPU 및 네트워크 과부하 | **10 FPS 정속 프레임 스킵 다운스케일링** | 연산 부하 및 대역폭 **67% 대폭 절감** |
| **정체시간 톱니파 진동** | 4초마다 정체시간이 0초~30초로 톱니처럼 급변 | **EMA 혼잡 누적/감쇠 모델 적용** | 2~8초 수준으로 매끄럽고 현실적인 정체 지표 유지 |
| **공중 노이즈 차단** | 천장 전광판 영상 속 인물을 보행자로 오인식 | **역마스킹 (Inverse Masking)** | 공중/전광판 오인식 **100% 원천 차단** |

---

## 📂 프로젝트 폴더 구조

```text
TDTC-AI-CCTV/
├── ai_server.py           # FastAPI 통합 서버 진입점 (포트 8088)
├── run_ai_server.ps1      # 원클릭 서버 실행 스크립트 (FastAPI + Cloudflare Tunnel)
├── requirements.txt       # 의존성 패키지 목록
├── .env                   # 환경변수 설정 파일 (RDS, S3, 고정 주소)
├── .gitignore             # 깃 형상관리 보안/임시파일 제외 설정
│
├── server/                # [핵심] 모듈화된 관제 서버 패키지
│   ├── routers/           # API 라우터 (cctv.py, analyze.py, alerts.py)
│   ├── config.py          # 환경 설정, 전역 상태, Pydantic DTO
│   ├── models.py          # CSRNet / YOLO11n 모델 및 BEV 좌표 변환기
│   ├── services.py        # 실시간 AI 파이프라인, 자동 알람 엔진, 백엔드 Webhook 연동
│   ├── tracker.py         # 객체 추적기, 정체시간 산출, 흔들림 보정
│   ├── raw_video_worker.py# 1분 단위 상시 영상 분할 및 S3 적재 워커
│   ├── video_buffer.py    # 35초 위험 클립 추출용 원형 프레임 버퍼
│   ├── pdf_generator.py   # 사고 명세서 PDF 생성기
│   ├── s3_uploader.py     # AWS S3 파일 업로더
│   └── websocket.py       # 실시간 WebSocket 연결 관리자
│
├── models/                # AI 모델 가중치 (.pth, yolo11n.pt)
├── utils/                 # PostgreSQL RDS 커넥터 유틸리티 (psycopg2)
└── results/               # 분석 결과 영상/PDF/JSON 임시 저장소
```

---

## 🚀 실행 가이드 (Getting Started)

### 1. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 실시간 통합 서버 실행
```powershell
# FastAPI (포트 8088) 및 Cloudflare 영구 고정 터널 동시 구동
.\run_ai_server.ps1
```

* **로컬 서버 주소**: `http://localhost:8088`
* **라이브 접속 주소 (Cloudflare)**: `https://tdtc-ai-cctv.uk`
* **Swagger UI (API 명세서)**: `https://tdtc-ai-cctv.uk/docs`
* **실시간 WebSocket 관제**: `wss://tdtc-ai-cctv.uk/ws/cctv-stream`

---

## 📡 주요 API 엔드포인트 명세

### 1. 실시간 관제 및 스트리밍 (WebSocket & CCTV)
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `WS` | `/ws/cctv-stream` | 실시간 3개 구역 동기화 프레임 추론 결과(인원수, CRI, `stagnation_sec`) 브로드캐스트 |
| `GET` | `/api/v1/cctv/stream?zone_id={id}` | 구역별(Zone 1, 2, 3) 10 FPS 실시간 모자이크 비식별화 MJPEG 스트림 |
| `POST` | `/api/v1/cctv/upload` | 비디오 파일 업로드 ➡️ 실시간 분석, 1분 상시 녹화, 자동 알람 모니터링 및 RDS 적재 |
| `GET` | `/api/v1/cctv/status` | 현재 실시간 파이프라인 가동 상태 조회 |
| `GET` | `/health` | 서버 및 CUDA GPU(RTX 4060) 가동 상태 확인 |

### 2. 긴급 알람 및 사후 처리 Webhook
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `POST` | `/api/analyze/confirm` | 출동 확정 시 35초 클립 + PDF 생성 + S3 업로드 + BE 웹훅 호출 |
| `POST` | `/api/analyze/snapshot` | 백엔드/프론트엔드 Webhook 지시용 35초 스냅샷 트리거 |
| `POST` | `/api/alerts/trigger` | Java 백엔드로 긴급 112/119 SMS 알람 전송 |
