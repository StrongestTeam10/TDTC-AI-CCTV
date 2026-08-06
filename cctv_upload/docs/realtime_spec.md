# [Task Request] CSRNet 기반 3개 구역 실시간 영상 처리 파이프라인 최적화 (Real-Time Pipeline Engineering)

## 🎯 목표
현재 VRAM OOM으로 인해 `max_workers=2`로 제약되어 발생하는 **모델링 대기 시간(Queueing Wait)을 0초로 단축**하고, 3개 구역(Zone 1, 2, 3) 동영상 스트림을 **완전 동시 시작 / ms 오차 범위 내 동시 종료**되는 실시간(Real-Time, 30+ FPS) 관제 파이프라인으로 개편한다.

---

## 📌 핵심 요구사항 (Key Requirements)

### 1. 단일 모델 인스턴스 공유 & Batch Inference (VRAM 절감 및 병렬화)
- **Problem**: 스레드별 모델 로드로 인한 VRAM 고갈(OOM) 및 구역 3 대기 지연 발생.
- **Solution**: 
  - GPU 상에 CSRNet 모델 인스턴스를 **단 1개만 로드**하여 메모리를 공유한다.
  - 3개 구역 스트림의 프레임을 **Thread-safe Queue**로 수집한 후, `batch_size=3` (`(3, C, H, W)`) 형태의 Mini-batch로 묶어 GPU 연산 1회로 통합 추론한다.
  - 이를 통해 `max_workers=3` 완전 동시 처리를 구현하여 대기 작업을 없앤다.

### 2. TensorRT FP16 엔진 변환 및 가속 (AI 추론 가속)
- **Problem**: PyTorch Native FP32 (`csrnet.eval()`) 추론 속도 한계.
- **Solution**: 
  - 기존 CSRNet 모델을 ONNX Export 후 **TensorRT FP16 (`.engine` / `.trt`)**으로 가속 변환한다.
  - Layer Fusion 및 Tensor Core 연산을 활용하여 추론 속도를 2~3배 이상 추가 가속한다.

### 3. 기존 고성능 파이프라인 하위 호환성 유지 (Preserve Existing Optimizations)
기존에 구축된 다음 5가지 핵심 기법은 그대로 유효하게 작동하도록 아키텍처를 구성한다:
1. **GPU MaxPool2d (Zero-Resize)**: AI 출력 히트맵($160 \times 90$) 텐서 상태에서 GPU 5x5 MaxPool2d 연산으로 좌표 직접 추출.
2. **10 FPS 적응형 샘플링**: Target FPS = 10.0 (매 3프레임 중 1프레임 분석).
3. **`cap.grab()` 포인터 스킵**: 건너뛰는 프레임은 `cap.grab()`으로 CPU H.264 디코딩 비용 제거.
4. **Producer-Consumer 비동기 파이프라인**: [프레임 디코딩 / Prefetch] ➔ [GPU Batch 추론] ➔ [좌표 추출 및 DB 적재] 스레드 분리.
5. **Server-Client 분산 렌더링**: 서버는 좌표 JSON 데이터만 DB 적재, 렌더링은 클라이언트 브라우저 Canvas/WebGL로 위임.

---

## 🏗️ 시스템 파이프라인 구조
[Zone 1 Stream] ──(cap.grab/read)──┐
[Zone 2 Stream] ──(cap.grab/read)──┼──> [Thread-safe Queue] ──> [TensorRT CSRNet (Shared 1 Instance)]
[Zone 3 Stream] ──(cap.grab/read)──┘      (Collects 3 frames)           (Batch Size = 3 Inference)
│
▼
[DB / API] <── [JSON Coordinates] <── [GPU 5x5 MaxPool2d NMS] <── [Density Maps Output]
---

## 🛠️ 구현 요청 상세 (Implementation Details)

1. **Queue & Batching Coordinator 작성**:
   - 3개 구역 비디오 디코더 스레드에서 프레임과 구역 ID를 비동기로 Queue에 Push한다.
   - Inference Worker 스레드는 Queue에서 3개 구역 프레임이 모이는 즉시 Tensor Stack (`torch.cat` 또는 `np.stack`) 후 TensorRT Engine에 전달한다.
   - Batch 추론 결과 Tensor를 구역별로 분할(Unbatch)하여 비동기로 MaxPool2d 좌표 추출 및 DB 적재 파이프라인으로 전달한다.

2. **TensorRT Inference Wrapper Class 작성**:
   - Dynamic Shape 또는 Fixed Batch Size = 3 환경에 맞춘 TensorRT Engine Binding 및 PyCUDA/TensorRT C++ Python API 인퍼런스 래퍼 클래스를 구현한다.

3. **Exception Handling & Thread Safety**:
   - 스트림 중단 시 타임아웃 처리 및 Queue deadlock 방지 로직 포함.
   - CUDA Context 스레드 바인딩 관리 (`cuda.Context.push() / pop()` 또는 PyTorch CUDA Stream 활용).

---

## 🏁 최종 검증 지표 (Success Metrics)
- **VRAM 사용량**: 기존 대비 1/3 수준 유지 (OOM 발생 0회).
- **구역간 실행 편차**: Zone 1, 2, 3 처리 완료 시점 간격 **< 10ms 이내 (동시 종료)**.
- **처리 속도**: 3개 구역 통합 처리 속도 **30+ FPS 이상 (실시간 달성)**.

---

## 🛡️ 프로덕션 안정성 및 엔지니어링 가이드라인 (Production Guidelines)

### 1. 30ms 타임아웃 동적 배치 (Dynamic Timeout Batching)
- **적용 이유**: 특정 구역 카메라/비디오의 지연 및 중단 발생 시 시스템 전체가 멈추는 Queue Deadlock 방지 (NVIDIA Triton Inference Server 설계 표준).
- **구현 방식**: 30ms 타임아웃 내 3개 프레임 미수집 시, 수집된 프레임만 우선 추론 진행.

### 2. Dummy Zero-Tensor Padding (고정 배치 수율 극대화)
- **적용 이유**: TensorRT Dynamic Shape 사용 시 발생하는 5~10% 커널 최적화 성능 저하 방지.
- **구현 방식**: `batch_size=3` 고정 엔진으로 빌드하고, 미달된 프레임 자리는 `torch.zeros_like()` 검은색 Dummy Tensor로 채워 추론 후 버림(Drop).

### 3. 단일 GPU 전담 스레드 (Single Dedicated GPU Worker)
- **적용 이유**: Windows 환경 멀티스레드 간 CUDA Context 스위칭 오류(`CUDA_ERROR_INVALID_CONTEXT`) 100% 차단.
- **구현 방식**: 디코딩 스레드(Zone 1, 2, 3)는 CPU 비디오 디코딩 후 Queue Push만 담당하고, GPU VRAM 접근 및 TensorRT 연산은 오직 **단 1개의 Dedicated GPU Worker 스레드**가 독점 처리.

### 4. FramePackage Dataclass 메타데이터 패킹
- **적용 이유**: Batch 추론 후 Unbatching 과정에서 3개 구역 프레임 간 순서 뒤섞임 방지.
- **구현 방식**: `(zone_id, frame_idx, timestamp, tensor)` 형태의 Dataclass 객체로 패킹하여 Queue로 전송.


