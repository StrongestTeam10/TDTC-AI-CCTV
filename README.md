# 📡 TDTC-AI-CCTV (CCTV AI 분석 및 데이터 적재 파이프라인)

본 저장소는 **망원시장 스마트 CCTV 인원 분석 및 인파 밀집도 위험 감지**를 수행하는 AI 파이프라인이자, 분석된 데이터를 Supabase(PostgreSQL)에 대량 적재하고 긴급 신고 이벤트를 Java 백엔드로 중계하는 **FastAPI API 서버**입니다.

---

## 🏗️ 시스템 아키텍처 및 데이터 흐름

```
[CCTV 스트림/업로드 비디오]
           │
           ▼
[FastAPI AI 분석 서버 (:8088)]
   │
   ├─ (1) CSRNet AI 모델 추론
   ├─ (2) 2D 픽셀 ➡️ 3D BEV 물리 좌표 변환 (Homography)
   ├─ (3) 글로벌 객체 추적 (Hungarian Matching & Kalman Filter)
   │
   ├─ (4) [대량 적재] psycopg2 Bulk Insert ───────────────┐
   │                                                     ▼
   └─ (5) [긴급 알람] HTTP POST ──→ [Java BE (:8080)] ──→ [Supabase DB]
                                       (SMS 발송)
```

---

## 📂 프로젝트 폴더 구조 및 파일 순서

```directory
TDTC-AI-CCTV/
├── ai_pipeline/
│   ├── cctv_upload/
│   │   ├── core/
│   │   │   ├── 01_frame_utils.py         # 영상 프레임/RTSP 스트림 캡처 헬퍼
│   │   │   ├── 02_inference.py           # CSRNet 모델 추론 엔진
│   │   │   ├── 03_postprocess.py         # 밀도맵 후처리 및 BEV 변환
│   │   │   ├── 04_db_client.py           # DB 접속 클라이언트
│   │   │   ├── 05_logger.py              # 파이프라인 로그 수집기
│   │   │   └── zones_config.json         # 호모그래피 행렬 및 관제 ROI 설정
│   │   └── steps/
│   │       ├── 04_video_to_bev_CSR.py    # [실행순서 1] 비디오 파일에서 BEV 물리 좌표 추출
│   │       └── 09_aggregate_pedestrian_json.py # [실행순서 2] 보행자 추적 및 DB 일괄 적재
│   │
│   ├── cctv_ai_pipeline/
│   │   └── sensor_fusion_archive/
│   │       └── utils/
│   │           └── db_connector.py       # Supabase PostgreSQL Bulk Insert 핵심 모듈
│   │
│   ├── .gitignore                        # 대용량 영상 및 가중치 파일 차단 필터
│   └── ai_server.py                      # FastAPI API 서버 엔트리 포인트 (Port: 8088)
│
└── run_ai_server.ps1                     # FastAPI + ngrok 자동 통합 구동 스크립트
```

---

## 🚀 실행 가이드

### 1️⃣ 가상환경 구축 및 패키지 설치
Python 3.10+ 환경 및 CUDA(NVIDIA GPU) 환경 구동을 권장합니다.
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate   # Windows 기준

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 환경 변수 설정 (`.env`)
`ai_pipeline/.env` 파일을 생성하고 아래 양식에 맞추어 정보를 입력합니다.
```env
# AWS S3 설정
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-northeast-2

# Supabase PostgreSQL 접속 정보
DB_HOST=aws-0-ap-northeast-1.pooler.supabase.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=your_db_username
DB_PASSWORD=your_db_password

# Java 백엔드 연동 설정
BACKEND_URL=http://localhost:8080
BACKEND_API_KEY=your_backend_secret_key
```

### 3️⃣ AI 서버 및 ngrok 터널링 자동 기동
루트 폴더에서 제공되는 파워쉘 스크립트를 사용하여 FastAPI 서버(`:8088`)와 외부 터널링 주소를 원클릭으로 구동합니다.
```powershell
.\run_ai_server.ps1
```
* **결과**: `https://scenic-dander-nuttiness.ngrok-free.dev` 고정 주소를 통해 외부 인터넷에서 로컬 AI 분석 서버로 안전하게 바인딩됩니다.

---

## 🔌 API 명세 요약

자세한 API 정의는 서버 구동 후 **[Swagger UI](https://scenic-dander-nuttiness.ngrok-free.dev/docs)**에서 확인할 수 있습니다.

### 1. 상태 검사
* **`GET /health`**
  * 서버 생존 상태 및 GPU 사용 가능 여부(`CUDA`)를 반환합니다.

### 2. CCTV 비디오 분석 요청 (비동기 백그라운드)
* **`POST /api/analyze/trigger`** (Multipart/Form-Data)
  * 사용자가 업로드한 `.mp4` 비디오를 임시 저장한 후, 백그라운드 태스크로 AI 분석 및 Supabase 적재 파이프라인을 기동합니다.
  * **Parameters**:
    * `file`: 비디오 파일
    * `zone_id`: 관제 구역 번호 (1: 남쪽, 2: 중앙, 3: 북쪽)
    * `fps`: 분석 시 샘플링할 목표 프레임률 (기본값: 10.0)

### 3. 진행 상황 폴링 조회
* **`GET /api/analyze/status`**
  * 현재 백그라운드에서 실행 중인 비디오 분석 파이프라인의 실시간 진행률(`progress_percent`, 0%~100%) 및 상태 메시지를 반환하여 화면상에 프로그레스 바를 그릴 수 있게 해줍니다.
