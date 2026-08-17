# 🎥 TDTC CCTV AI 통합 분석 서버 2.1 (TDTC-AI-CCTV)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![AWS S3](https://img.shields.io/badge/AWS%20S3-tdtc--cctv--upload-FF9900?style=for-the-badge&logo=amazons3)](https://aws.amazon.com/s3)
[![Swagger Docs](https://img.shields.io/badge/Swagger%20UI-Live%20Docs-0055FF?style=for-the-badge&logo=swagger)](https://scenic-dander-nuttiness.ngrok-free.dev/docs)

망원시장 스마트 CCTV의 다중 구역(Zone 1, 2, 3)을 관제하는 **실시간 AI 비디오 분석, 정체시간 산출, S3 분산 스토리지 적재, 긴급 자동/수동 알람 웹훅 중계 서버**입니다.

---

## 🌟 핵심 아키텍처 및 기능 (Core Features)

### 1. 🤖 CSRNet + YOLO 앙상블 군중 분석 & 정체 시간 산출
* **CSRNet**: 군중 밀도 지도(Density Map) 추론 및 보행자 정밀 머리 중심점(Peak) 픽셀 좌표 추출
* **가우시안 모자이크(비식별화)**: 개인정보 보호를 위한 보행자 얼굴/신체 영역 실시간 블러 처리
* **BEV (Bird's Eye View) 물리 좌표 변환**: Homography ($H$-Matrix) 기반으로 CCTV 2D 픽셀을 실세계 미터 단위 3D 평면 좌표계로 역투영
* **정체 시간(`stagnation_sec`) 실시간 계산**: 칼만 필터 및 ID 궤적 이동 거리(Displacement)를 분석하여 인파 정체 및 병목 현상을 실시간 수치화 후 CRI 점수에 가중 반영

### 2. 📡 실시간 다중 구역 5 FPS WebSocket 최적화
* **WebSocket 엔드포인트**: `/ws/cctv-stream`
* **5 FPS Throttling**: 프론트엔드 브라우저 부하를 최소화하기 위해 0.2초(5 FPS) 간격으로 프레임 지표 브로드캐스트
* **구역별 독립 푸시**: Zone 1(남측), Zone 2(중앙), Zone 3(북측) 관제 데이터에 `zone_id`를 보장하여 대시보드에서 독립적인 오버레이/차트 렌더링 지원

### 3. 🚨 실시간 자동 위험 감지 및 3분 쿨다운 엔진 (Auto-Alert)
* **위험 감지 기준**: CRI 점수 70pt 이상 위험 상황이 **10초간 연속 지속**될 때 AI 서버가 스스로 판단
* **비동기 자동 발령**: 35초 위험 클립 추출 + 사고 명세서 PDF 생성 + S3 업로드 + Java 백엔드 `alertType: "AUTO_REPORT"` 웹훅 및 112/119 긴급 SMS 발송
* **쿨다운 타이머**: 동일 구역 내 3분(180초) 동안 중복 알람이 발송되지 않도록 쿨다운 타이머 제어

### 4. 📼 35초 위험 상황 클립(Warning Clip) 원형 버퍼링
* **Rolling Frame Buffer (`video_buffer.py`)**: 메모리 상의 원형 큐(`deque`)로 최근 35~40초간의 프레임을 상시 보관
* 위험 감지 또는 프론트엔드 출동 버튼 클릭 시 즉시 전후 35초 분량 MP4 클립 추출
* S3 `warning-clips/` 폴더에 업로드 후 **`clipType: "RISK"`** 로 백엔드 DB에 영구 보존 등록

### 5. ✂️ 1분 단위 상시 원본 영상(Raw Video) 스플릿 워커
* **Raw Video Splitter (`raw_video_worker.py`)**: 실시간 추론 스트림 프레임을 1분(60초) 단위로 자동 분할 인코딩
* S3 `raw-videos/` 폴더에 업로드 후 **`clipType: "TEMP"`** 로 백엔드 DB 등록 (백엔드 스케줄러가 1시간 후 자동 청소하여 스토리지 용량 최적화)

### 6. 📄 정형 사고 명세서 PDF 자동 생성
* **PDF Generator (`pdf_generator.py`)**: 사고 발생 시간, Zone ID, 위험 유형, CRI Score, 보행자 수/점유율, 정체 시간, 상황 요약, CCTV 캡처 스냅샷 이미지가 포함된 정식 PDF 명세서 렌더링
* S3 `report-pdfs/` 업로드 및 백엔드 `POST /api/v1/post-reports` 웹훅 전송

---

## 🛠️ 핵심 성능 최적화 및 트러블슈팅 (Optimization)

| 해결 과제 | 기존 문제점 | 적용 해결 기법 | 최종 개선 성과 |
| :--- | :--- | :--- | :--- |
| **공중 노이즈 차단** | 천장 전광판 영상 속 인물 및 조명을 보행자로 오인식 | **역마스킹 (Inverse Masking)**<br>보행 유효 구역 외곽을 0(검은색) 처리 후 모델 전달 | 공중/전광판 오인식 **100% 원천 차단** |
| **원근 왜곡 보정** | 탑뷰 CCTV 시점에 따른 머리 위치와 바닥 구역 불일치 | **Y축 오프셋 보정 (Head-to-Ground)**<br>머리 검출 좌표에 원근법 수식 적용 지면으로 하향 보정 | Zone 출입 판정 정밀도 극대화 |
| **중복 카운팅 해결** | 장애물 가림 시 1~2프레임마다 불필요한 신규 ID 생성 | **트래킹 유예 (MAX_LOST_FRAMES=5)**<br>칼만 필터 기반 최대 0.5초간 추적 상태 유지 | 중복 ID 생성 **90% 이상 대폭 감소** |
| **인파 병목/정체 감지** | 단순 인원수만 측정하여 서행/정체 위험 감지 불가 | **변위 기반 정체시간 산출 (`stagnation_sec`)**<br>객체별 이동 거리 $\Delta d < 8px$ 누적으로 체류시간 분석 | CRI 위험도 지표에 정체 가중치 반영 |
| **알람 폭발 방지** | 위험 프레임마다 알람이 발생하여 SMS 폭주 위험 | **지속 감지(10초) & 3분 쿨다운 타이머**<br>구역별 독립적인 쿨다운 맵으로 중복 알람 차단 | 비상 문자 중복 발송 **100% 방지** |
| **실시간 추론 속도** | FP32 연산으로 인한 프레임 처리 지연 | **PyTorch GPU FP16 가속 (AMP)**<br>`torch.amp.autocast('cuda')` 최신 구문 적용 | 처리 지연시간 **200ms 이하 유지** |

---

## 📂 프로젝트 폴더 구조

```text
TDTC-AI-CCTV/
├── ai_server.py           # FastAPI 통합 서버 진입점 (포트 8088)
├── run_ai_server.ps1      # 원클릭 서버 실행 스크립트 (FastAPI + ngrok)
├── requirements.txt       # 의존성 패키지 목록
├── .env                   # 환경변수 설정 파일
├── .gitignore             # 깃 형상관리 보안/임시파일 제외 설정
│
├── server/                # [핵심] 모듈화된 관제 서버 패키지
│   ├── routers/           # API 라우터 (cctv.py, analyze.py, alerts.py)
│   ├── config.py          # 환경 설정, 전역 상태, Pydantic DTO
│   ├── models.py          # CSRNet 신경망 모델 및 BEV 좌표 변환기
│   ├── services.py        # 실시간 AI 파이프라인, 자동 알람 엔진, 백엔드 Webhook 연동
│   ├── tracker.py         # 객체 추적기, 정체시간(stagnation_sec) 산출, 흔들림 보정
│   ├── raw_video_worker.py# 1분 단위 상시 영상 분할 및 S3(TEMP) 적재 워커
│   ├── video_buffer.py    # 35초 위험 클립 추출용 원형 프레임 버퍼
│   ├── pdf_generator.py   # 사고 명세서 PDF 생성기
│   ├── s3_uploader.py     # AWS S3 파일 업로더 (시뮬레이션 Fallback 지원)
│   └── websocket.py       # 실시간 WebSocket 연결 관리자
│
├── models/                # AI 모델 가중치 (.pth, .pt)
├── utils/                 # DB 커넥터 유틸리티 (psycopg2)
└── results/               # 분석 결과 영상/PDF/JSON 임시 저장소
```

*(참고: 일회성 DB 초기화 및 검증용 배치 스크립트는 최상위 `cctv_batch_tools/` 폴더로 분리 관리됩니다.)*

---

## 🚀 실행 가이드 (Getting Started)

### 1. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 실시간 통합 서버 실행
```powershell
# FastAPI (포트 8088) 및 ngrok 고정 도메인 터널링 구동
.\run_ai_server.ps1
```

* **로컬 서버 주소**: `http://localhost:8088`
* **라이브 접속 주소 (ngrok)**: `https://scenic-dander-nuttiness.ngrok-free.dev`
* **Swagger UI (API 명세서)**: `https://scenic-dander-nuttiness.ngrok-free.dev/docs`

---

## 📡 주요 API 엔드포인트 명세

### 1. 실시간 관제 및 스트리밍 (WebSocket & CCTV)
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `WS` | `/ws/cctv-stream` | 실시간 프레임 추론 결과(인원수, CRI, `stagnation_sec`, 좌표) 5 FPS 브로드캐스트 |
| `POST` | `/api/v1/cctv/upload` | 비디오 파일 업로드 ➡️ 실시간 분석, 1분 상시 녹화, 자동 알람 모니터링 및 DB 적재 |
| `GET` | `/api/v1/cctv/status` | 현재 실시간 파이프라인 가동 상태 조회 |
| `GET` | `/api/v1/cctv/video/{filename}` | 모자이크 처리된 결과 영상 다운로드 및 스트리밍 반환 |
| `GET` | `/api/v1/cctv/dataset/{filename}` | 분석 데이터셋 JSON 다운로드 |

### 2. 긴급 알람 및 사후 처리 Webhook
| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| `POST` | `/api/analyze/confirm` | 출동 확정 시 35초 클립 + PDF 생성 + S3 업로드 + BE 웹훅 호출 |
| `POST` | `/api/analyze/snapshot` | 백엔드/프론트엔드 Webhook 지시용 35초 스냅샷 트리거 |
| `POST` | `/api/alerts/trigger` | Java 백엔드(`/api/ai/alerts/trigger`)로 긴급 112/119 SMS 알람 전송 |
| `GET` | `/health` | 서버 및 CUDA GPU 상태 확인 |

---

## 🚨 클라우드 S3 및 백엔드 연동 규격

### 1. 영상 클립 타입 (`clipType`) 엄격 분리
* **상시 1분 영상 (Raw Video)**: `POST /api/v1/video-clips` 에 `"clipType": "TEMP"` 로 전송 ➡️ **1시간 후 자동 삭제(용량 최적화)**
* **위험 35초 영상 (Warning Clip)**: `POST /api/v1/video-clips` 에 `"clipType": "RISK"` 로 전송 ➡️ **사고 증거 자료로 영구 보존**

### 2. S3 버킷 디렉터리 구조 (`tdtc-cctv-upload`)
* `raw-videos/`: 1분 단위 상시 녹화 분할 MP4
* `warning-clips/`: 35초 위험 상황 전후 추출 MP4
* `report-pdfs/`: 사고 상세 분석 PDF 명세서
