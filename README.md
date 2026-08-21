# 🎥 TDTC CCTV AI 통합 분석 서버 (TDTC-AI-CCTV)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![YOLO11](https://img.shields.io/badge/YOLO-11n-00FFFF?style=for-the-badge&logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![Cloudflare Tunnel](https://img.shields.io/badge/Cloudflare-Permanent%20Domain-F38020?style=for-the-badge&logo=cloudflare)](https://tdtc-ai-cctv.uk)
[![Swagger Docs](https://img.shields.io/badge/Swagger%20UI-Live%20Docs-0055FF?style=for-the-badge&logo=swagger)](https://tdtc-ai-cctv.uk/docs)

망원시장 스마트 CCTV 다중 구역(Zone 1 남측, Zone 2 중앙, Zone 3 북측)을 관제하는 **실시간 30 FPS 비전 AI 추론(모자이크 비식별화), 10 FPS 1분 완주 모자이크 정기 녹화 및 S3/RDS 벌크 적재, 35초 비상 알람 Webhook 중계 서버**입니다.

---

## 🌟 핵심 아키텍처 및 2-Track 독립 파이프라인

```mermaid
flowchart TD
    subgraph AI_Core ["🐍 CCTV AI 서버 (FastAPI :8088)"]
        YOLO["YOLO11n (보행자 검출 & 비식별화 모자이크)"]
        CSRNET["PyTorch CSRNet (인파 밀집도 & 3D 좌표)"]
        RENDERER["독립 백그라운드 1분 모자이크 렌더러 (10 FPS)"]
        BUF["35초 원형 롤링 버퍼 (비상 클립)"]
    end

    subgraph Hot_Track ["⚡ Track 1: 실시간 라이브 관제 (30 FPS 고속 송출)"]
        CF_TUNNEL["🚇 Cloudflare 고정 터널 (https://tdtc-ai-cctv.uk)"]
        FE_LIVE["⚛️ 웹 관제 대시보드 (MJPEG 30 FPS 스트림 & WebSocket 지표)"]
        AI_Core ==> CF_TUNNEL ==> FE_LIVE
    end

    subgraph Cold_Track ["🗄️ Track 2: 클라우드 아카이빙 및 RDS 5대 테이블 적재"]
        direction TB
        %% 1분 영상 & 메트릭 벌크 적재
        RENDERER -->|1. 1분 완주 모자이크 mp4 업로드| S3_RAW[("☁️ AWS S3 (raw-videos/)")]
        RENDERER -->|2. POST /api/v1/video-clips| BE_CLIP["☕ Spring Boot 백엔드"]
        RENDERER -->|3. POST /api/v1/metrics/bulk (600프레임 일괄 적재)| BE_BULK["☕ Spring Boot 백엔드"]
        
        BE_CLIP -->|vdoclip01m 저장| RDS[(🗄️ AWS RDS PostgreSQL)]
        BE_BULK -->|pedaggr01h / mrkrisk01m 저장| RDS

        %% 비상 위험 클립
        BUF -->|위험 감지 시 35초 mp4 업로드| S3_DANGER[("☁️ AWS S3 (danger-clips/)")]
        BUF -->|POST /api/ai/alerts/trigger| BE_ALERT["☕ Spring Boot 백엔드"]
        BE_ALERT -->|emgalrt01h / pstrprt01h 저장 & SMS 발송| RDS
    end
```

---

## 🚀 빠른 시작 (Quick Start)

파이썬 AI 서버와 Cloudflare 영구 터널(`https://tdtc-ai-cctv.uk`)을 원클릭으로 실행합니다:

```powershell
# 1. 운영(PROD) 모드로 실행 (기본 권장)
.\run_ai_server.ps1

# 2. 로컬 개발(DEV) 모드로 실행
.\run_ai_server.ps1 dev
```

---

## 🏛️ AI CCTV 핵심 5대 테이블 적재 파이프라인

| 번호 | 데이터베이스 테이블 | 관리 데이터 | 적재 주기 & 방식 |
| :---: | :--- | :--- | :--- |
| **1** | **`vdoclip01m`** | **1분 상시 녹화 및 35초 비상 영상 S3 URL 메타데이터** | 1분 완료 시 `POST /api/v1/video-clips` |
| **2** | **`pedaggr01h`** | **보행자 2D 픽셀 및 3D BEV 물리 좌표 (프레임별)** | 1분 완료 시 `POST /api/v1/metrics/bulk` |
| **3** | **`mrkrisk01m`** | **구역별 위험도 지표 (인원수, 밀집도, 정체시간, CRI 점수)** | 1분 완료 시 `POST /api/v1/metrics/bulk` |
| **4** | **`emgalrt01h`** | **비상 경보 및 SMS 발송 이력 (DANGER 레벨)** | 비상 감지 시 `POST /api/ai/alerts/trigger` |
| **5** | **`extfctr01h`** | **기상청 외부 환경 요인 (기온, 강수량, 습도 등)** | 기상청 날씨 API 연동 배치 |

---

## 📂 프로젝트 디렉토리 구조

```text
TDTC-AI-CCTV/
├── ai_server.py           # FastAPI 통합 서버 진입점 (포트 8088)
├── run_ai_server.ps1      # 프로필 기반 원클릭 실행 스크립트 (PROD / DEV)
├── requirements.txt       # 의존성 패키지 목록
├── .env.prod              # 운영용 환경변수 (AWS RDS & S3 & CloudFront)
├── .env.dev               # 개발용 환경변수 (Supabase DB)
├── .gitignore             # 보안 키, 모델 가중치, 미디어 파일 차단 규칙
│
├── server/                # [핵심] 모듈화된 관제 서버 패키지
│   ├── routers/           # API 라우터 (cctv.py, analyze.py, alerts.py)
│   ├── config.py          # 프로필 동적 로딩, 환경 설정, Pydantic DTO
│   ├── models.py          # CSRNet / YOLO11n 모델 및 BEV 좌표 변환기
│   ├── services.py        # 실시간 AI 파이프라인, 자동 알람 엔진, 백엔드 Webhook 연동
│   ├── tracker.py         # 객체 추적기, 정체시간 산출, 흔들림 보정
│   ├── raw_video_worker.py# 1분 단위 완주 모자이크 비디오 렌더러 및 벌크 적재 워커
│   ├── video_buffer.py    # 35초 위험 클립 추출용 원형 프레임 버퍼
│   ├── pdf_generator.py   # 사고 명세서 PDF 생성기
│   ├── s3_uploader.py     # AWS S3 파일 업로더
│   └── websocket.py       # 실시간 WebSocket 연결 관리자 (닫힌 소켓 자동 정리)
│
├── models/                # AI 모델 가중치 (.pth, bestYOLOm5080model.pt)
└── results/               # 분석 결과 임시 저장소
```

---

## 📡 주요 API 엔드포인트 명세

### 1. 실시간 관제 및 스트리밍 (WebSocket & CCTV)
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `WS` | `/ws/cctv-stream` | 실시간 3개 구역 동기화 프레임 추론 결과(인원수, CRI, 정체시간) 초당 5회 푸시 |
| `GET` | `/api/v1/cctv/stream?zone_id={id}` | 구역별(Zone 1, 2, 3) 30 FPS 실시간 모자이크 비식별화 MJPEG 스트림 |
| `POST` | `/api/analyze/trigger` | 영상 파일 수동 업로드 분석 트리거 |
| `GET` | `/health` | 서버 및 CUDA GPU(RTX 4060) 가동 상태 확인 |

### 2. 긴급 알람 및 사후 처리 Webhook
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `POST` | `/api/analyze/confirm` | 출동 확정 시 35초 클립 + PDF 생성 + S3 업로드 + BE 웹훅 호출 |
| `POST` | `/api/alerts/trigger` | Java 백엔드로 긴급 112/119 SMS 알람 전송 |
