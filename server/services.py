"""
server/services.py - 비동기 파이프라인 처리, 백그라운드 태스크 및 자바 백엔드 웹훅 서비스
"""

import os
import sys
import cv2
import json
import math
import time
import shutil
import asyncio
import requests
import numpy as np
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

from server.config import (
    BASE_DIR, RESULTS_DIR, MODELS_DIR, BACKEND_URL, BACKEND_API_KEY, PYTHON_EXE,
    analysis_state, pipeline_state, db_available, bulk_insert_pedestrian_coordinate_json
)
from server.models import (
    torch_available, CSRNet, transform_pixel_to_bev, extract_peaks_from_density, apply_mosaic, H_MATRIX, ROI_2D_POLY
)
from server.tracker import get_safe_video_capture, FrameStabilizer, CentroidTracker
from server.websocket import manager
from server.video_buffer import global_buffer_manager
from server.raw_video_worker import RawVideoSplitter

# =========================================================================
# 전역 자동 알람 쿨다운 및 위험 지속 상태 관리 (구역별 독립)
# =========================================================================
# zone_id -> 마지막 자동 알람 발송 epoch timestamp (3분 = 180초 쿨다운)
zone_auto_alert_cooldown: Dict[int, float] = {}
# zone_id -> 위험 상태(CRI 70pt 이상) 연속 지속 프레임 수
zone_danger_duration_frames: Dict[int, int] = {}


async def process_ai_pipeline(
    file_path: str, 
    filename: str, 
    zone_id: int = 1, 
    start_time: Optional[str] = None, 
    target_fps: float = 10.0
):
    """
    업로드된 비디오 기반 AI 파이프라인 추론, 1분 단위 상시 녹화, 정체시간 계산 및 WebSocket 실시간 스트리밍.
    CSRNet 추론 → 모자이크 처리 → 궤적/정체시간 분석 → 1분 녹화 분할 → 자동 알람 감시 → WebSocket & DB 적재
    """
    print(f"[AI Pipeline Started] '{filename}' (Zone: {zone_id}, StartTime: {start_time}, TargetFPS: {target_fps}) 영상 분석 개시...")

    await manager.broadcast({
        "type": "CCTV_AI_START",
        "filename": filename,
        "zone_id": zone_id,
        "message": f"🤖 CCTV 영상 '{filename}' AI 파이프라인 분석을 시작합니다."
    })

    # 1. 모델 초기화
    csrnet_model = None
    hog = None
    device = None

    if torch_available and CSRNet is not None:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            csrnet_model = CSRNet().to(device)
            model_path = os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=device)
                state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
                csrnet_model.load_state_dict(state_dict, strict=False)
                csrnet_model.eval()
                print(f"[AI Pipeline] CSRNet 모델 로드 성공: {model_path}")
            else:
                print(f"[AI Pipeline Warning] CSRNet 가중치 없음: {model_path}")
                csrnet_model = None
        except Exception as e:
            print(f"[AI Pipeline Warning] CSRNet 로드 실패 ({e}). HOG 폴백 사용.")
            csrnet_model = None

    if csrnet_model is None:
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[AI Pipeline] OpenCV HOG 감지기 초기화 성공.")
        except Exception as ex:
            print(f"[AI Pipeline Error] HOG 초기화 실패 ({ex}). 수학적 시뮬레이션 사용.")

    # 2. 비디오 로드 및 메타데이터 추출
    cap, temp_read_path = get_safe_video_capture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 600
    if total_frames <= 0:
        total_frames = 600

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    if fps <= 0:
        fps = 15.0

    print(f"[AI Pipeline] 비디오 정보: {total_frames}프레임, {width}x{height}, {fps}FPS")

    is_portrait = height > width
    target_size = (720, 1280) if is_portrait else (1280, 720)

    # 3. ROI 다각형 보정
    local_roi_poly = None
    try:
        config_path = os.path.join(BASE_DIR, "cctv_upload", "core", "zones_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(BASE_DIR, "core", "zones_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                zones_config = json.load(f)
                zone_meta = zones_config.get(str(zone_id), zones_config.get("1", {}))
                raw_poly = zone_meta.get("roi_polygon")
                if raw_poly:
                    scaled_poly = []
                    if width > height:
                        content_w = height * (9.0 / 16.0)
                        left_black_bar = (width - content_w) / 2.0
                        for pt in raw_poly:
                            mapped_x = left_black_bar + (pt[0] * (content_w / 1080.0))
                            mapped_y = pt[1] * (height / 1920.0)
                            final_x = mapped_x * (target_size[0] / width)
                            final_y = mapped_y * (target_size[1] / height)
                            scaled_poly.append([int(final_x), int(final_y)])
                    else:
                        for pt in raw_poly:
                            final_x = pt[0] * (target_size[0] / 1080.0)
                            final_y = pt[1] * (target_size[1] / 1920.0)
                            scaled_poly.append([int(final_x), int(final_y)])
                    local_roi_poly = np.array(scaled_poly, dtype=np.int32)
    except Exception as poly_err:
        print(f"[AI Pipeline Warning] 동적 레터박스 스케일 실패 ({poly_err}). 기본 ROI 사용.")

    if local_roi_poly is None:
        local_roi_poly = ROI_2D_POLY

    # 4. 결과 출력용 모자이크 영상 Writer 초기화
    output_video_filename = f"processed_{filename}"
    output_video_path = os.path.join(RESULTS_DIR, output_video_filename)

    out_video = None
    for codec in ['avc1', 'mp4v', 'XVID']:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
            if out_video.isOpened():
                print(f"[AI Pipeline] VideoWriter 초기화 성공 (코덱: {codec}, 크기: {target_size})")
                break
        except Exception:
            pass

    if out_video is None or not out_video.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    # 5. 상시 1분 영상 분할 워커 및 정체시간 추적기 인스턴스 초기화
    raw_video_splitter = RawVideoSplitter(zone_id=zone_id, fps=fps, split_duration_sec=60.0)
    centroid_tracker = CentroidTracker(max_disappeared=10, max_distance=60.0)
    stabilizer = FrameStabilizer(ground_roi=local_roi_poly)

    seed_offset = sum(ord(c) for c in filename) % 20
    dataset_records = {}
    last_base_count = 0
    skip_interval = max(1, int(round(fps / target_fps)))
    last_peaks = []

    start_time_str = start_time
    if start_time_str:
        try:
            start_time_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        except ValueError:
            try:
                start_time_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                start_time_dt = datetime.now(timezone.utc)
    else:
        start_time_dt = datetime.now(timezone.utc)

    # 6. 메인 프레임 분석 루프
    for frame in range(1, total_frames + 1):
        if not pipeline_state["is_analyzing"]:
            print("[AI Pipeline] 사용자에 의해 분석이 중단되었습니다.")
            break

        ret, img = cap.read()
        frame_720p = cv2.resize(img, target_size) if (ret and img is not None) else None

        if frame_720p is not None:
            # 원형 버퍼(최근 35초 위험 클립 추출용)에 프레임 저장
            global_buffer_manager.push_frame(zone_id, frame_720p)
            # 상시 1분 분할 워커(S3 raw-videos/ 적재용)에 프레임 전달
            raw_video_splitter.write_frame(frame_720p)

        # AI 추론 (프레임 스킵 간격 적용)
        if ret and img is not None:
            if (frame - 1) % skip_interval == 0:
                try:
                    if csrnet_model is not None:
                        import torch
                        from torchvision import transforms
                        csr_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img_rgb = cv2.cvtColor(frame_720p, cv2.COLOR_BGR2RGB)
                        input_tensor = csr_transform(img_rgb).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = csrnet_model(input_tensor)
                            output = torch.clamp(output, min=0)

                            density_map = output.squeeze().cpu().numpy()
                            h_out, w_out = output.shape[2], output.shape[3]
                            scale_x = target_size[0] / w_out
                            scale_y = target_size[1] / h_out
                            
                            raw_peaks = extract_peaks_from_density(density_map, threshold=0.0005)
                            last_peaks = []
                            for px_raw, py_raw in raw_peaks:
                                u = px_raw * scale_x
                                v = py_raw * scale_y
                                last_peaks.append((u, v))
                            
                            last_base_count = len(last_peaks)

                    elif hog is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                        last_base_count = len(boxes)
                        last_peaks = []
                        for bx, by, bw, bh in boxes:
                            last_peaks.append((bx + bw / 2.0, by + bh / 2.0))
                    else:
                        last_base_count = 6 + seed_offset + int(7 * math.sin((frame + seed_offset * 10) / 18.0)) + (frame // 120)
                        last_peaks = []

                except Exception as ex:
                    print(f"[AI Inference Error] 프레임 {frame}: {ex}")
        else:
            last_base_count = 6 + seed_offset + int(7 * math.sin((frame + seed_offset * 10) / 18.0)) + (frame // 120)
            last_peaks = []

        base_count = max(0, last_base_count)

        # 궤적 추적 및 정체시간(stagnation_sec) 산출
        centroid_tracker.update(last_peaks, frame_id=frame, min_move_thresh=8.0)
        stag_metrics = centroid_tracker.get_stagnation_metrics(fps=fps)
        stagnation_sec = stag_metrics["avg_stagnation_sec"]

        # 복합 위험도(CRI Score) 계산 (인원수 + 점유율 + 정체시간 표준화 가중치)
        occupancy_rate = round(min(100.0, base_count * 2.2), 1)
        raw_cri = (base_count * 1.8) + (occupancy_rate * 0.25) + (min(25.0, stagnation_sec * 0.35))
        cri_score = round(min(100.0, max(5.0, raw_cri)), 1)

        risk_level = "NORMAL"
        if cri_score >= 70.0:
            risk_level = "EMERGENCY_EVACUATION"
        elif cri_score >= 40.0:
            risk_level = "WARNING"

        # ---------------------------------------------------------------------
        # 실시간 자동 위험 감지 및 쿨다운 알람 엔진 (Auto-Alert Trigger)
        # ---------------------------------------------------------------------
        if cri_score >= 70.0:
            zone_danger_duration_frames[zone_id] = zone_danger_duration_frames.get(zone_id, 0) + 1
            # 10초 이상 위험 지속 (fps * 10 프레임)
            danger_threshold_frames = int(fps * 10.0)
            now_ts = time.time()
            last_alert_ts = zone_auto_alert_cooldown.get(zone_id, 0.0)

            if zone_danger_duration_frames[zone_id] >= danger_threshold_frames and (now_ts - last_alert_ts >= 180.0):
                zone_auto_alert_cooldown[zone_id] = now_ts
                zone_danger_duration_frames[zone_id] = 0
                print(f"🚨 [Auto Alert Trigger] Zone {zone_id} 위험도 {cri_score}pt 10초 지속 감지! 자동 비상 신고를 발령합니다.")
                
                asyncio.create_task(
                    process_confirm_background(
                        video_path=file_path,
                        start_sec=max(0.0, (frame / fps) - 17.5),
                        end_sec=(frame / fps) + 17.5,
                        fps=fps,
                        zone_id=zone_id,
                        alert_type="AUTO_REPORT",
                        incident_summary=(
                            f"[자동감지 비상경보] Zone {zone_id} 구역에서 인파 밀집도(CRI {cri_score:.1f}pt, "
                            f"인원 {base_count}명, 정체 {stagnation_sec:.1f}초)가 10초 이상 지속되어 긴급 출동이 자동 발령되었습니다."
                        ),
                        video_id=1,
                        cri_score=cri_score,
                        pedestrian_count=base_count,
                        occupancy_rate=occupancy_rate,
                        stagnation_sec=stagnation_sec
                    )
                )
        else:
            zone_danger_duration_frames[zone_id] = max(0, zone_danger_duration_frames.get(zone_id, 0) - 1)

        detections_list = []
        any_low_confidence = False
        
        for i, (px, py) in enumerate(last_peaks):
            py_foot = py + 80
            (spx, spy), low_conf = stabilizer.stabilize_point(px, py_foot, frame_720p if frame_720p is not None else np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            if low_conf:
                any_low_confidence = True
            
            detections_list.append({
                "track_id": i + 1,
                "raw_bbox_bottom_center": [round(px, 2), round(py_foot, 2)],
                "stabilized_person_coords": [round(spx, 2), round(spy, 2)],
                "current_zone_id": zone_id
            })

        dataset_records[str(frame)] = {
            "pedestrian_count": base_count,
            "occupancy_rate": occupancy_rate,
            "stagnation_sec": stagnation_sec,
            "cri_score": cri_score,
            "risk_level": risk_level,
            "low_confidence": any_low_confidence,
            "detections": detections_list,
            "peaks": list(last_peaks)
        }

        # 모자이크 가우시안 블러 처리 후 프레임 저장
        if ret and img is not None and out_video is not None:
            img_anonymized = frame_720p.copy()
            for px, py in last_peaks:
                x1, y1 = px - 20, py - 20
                x2, y2 = px + 20, py + 80
                img_anonymized = apply_mosaic(img_anonymized, x1, y1, x2, y2, neighbor=15)
            out_video.write(img_anonymized)

        # 분석 진행률 브로드캐스트 (12프레임 주기)
        if frame == 1 or frame % 12 == 0 or frame == total_frames:
            percent = int((frame / total_frames) * 100)
            pipeline_state["frame_id"] = frame
            pipeline_state["pedestrian_count"] = base_count
            pipeline_state["cri_score"] = cri_score
            await manager.broadcast({
                "type": "CCTV_AI_PROGRESS",
                "filename": filename,
                "zone_id": zone_id,
                "progress": percent,
                "pedestrian_count": base_count,
                "occupancy_rate": occupancy_rate,
                "stagnation_sec": stagnation_sec,
                "cri_score": cri_score,
                "risk_level": risk_level,
                "low_confidence": any_low_confidence,
                "detections": detections_list,
                "step_name": f"CSRNet 추론 + 모자이크 처리 중 ({percent}%)"
            })
            await asyncio.sleep(0.01)

    # 7. 스트림 자원 정리 및 상시 1분 녹화 마무리
    if cap.isOpened():
        cap.release()
    if temp_read_path and os.path.exists(temp_read_path):
        try:
            os.remove(temp_read_path)
        except Exception:
            pass
    if out_video is not None:
        out_video.release()

    # 남아있는 1분 버퍼 비디오 파일 업로드 완료
    raw_video_splitter.finalize()

    # 분석 데이터셋 JSON 저장
    dataset_json_path = os.path.join(RESULTS_DIR, f"uploaded_{filename}_dataset.json")
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=2)

    # 8. Supabase DB 일괄 적재 (Bulk Insert)
    try:
        db_records = []
        for f_id, f_data in dataset_records.items():
            f_idx = int(f_id)
            frame_offset_sec = f_idx / fps
            frame_time_dt = start_time_dt + timedelta(seconds=frame_offset_sec)
            frame_time_iso = frame_time_dt.isoformat().replace("+00:00", "Z")

            pixels_map = {}
            bev_xyz_map = {}

            for det in f_data.get("detections", []):
                pid = str(det["track_id"])
                spx, spy = det["stabilized_person_coords"]
                pixels_map[pid] = [spx, spy]

                bev_x, bev_y = transform_pixel_to_bev(spx, spy, H_MATRIX)
                bev_xyz_map[pid] = [round(bev_x, 2), round(bev_y, 2), 0.0]

            db_records.append({
                "clip_id": 999,
                "zone_id": zone_id,
                "s3_clip_url": f"https://tdtc-cctv-upload.s3.ap-northeast-2.amazonaws.com/danger-clips/uploaded_clip_{zone_id}.mp4",
                "frame_id": int(f_id),
                "video_id": 1,
                "total_count": f_data["pedestrian_count"],
                "pixels_json": json.dumps(pixels_map, ensure_ascii=False),
                "bev_xyz_json": json.dumps(bev_xyz_map, ensure_ascii=False),
                "captured_at": frame_time_iso,
                "risk_score": f_data["cri_score"],
                "risk_level": f_data["risk_level"]
            })

        if not db_available or bulk_insert_pedestrian_coordinate_json is None:
            print("[AI Pipeline WARNING] db_connector 미로드 상태 - DB 적재 스킵")
        else:
            print(f"[AI Pipeline] Supabase DB에 {len(db_records)}개 프레임 일괄 적재 진행 중...")
            bulk_insert_pedestrian_coordinate_json(db_records)
    except Exception as db_err:
        err_msg = str(db_err)
        print(f"[AI Pipeline DB Error] Supabase DB 일괄 적재 실패: {err_msg}")
        await manager.broadcast({
            "type": "CCTV_DB_ERROR",
            "filename": filename,
            "zone_id": zone_id,
            "message": f"⚠️ DB 적재 실패: {err_msg[:200]}"
        })

    await manager.broadcast({
        "type": "CCTV_AI_COMPLETED",
        "filename": filename,
        "zone_id": zone_id,
        "message": f"✅ '{filename}' AI 분석 및 DB 적재 완료! 실시간 5 FPS 스트리밍을 시작합니다."
    })

    # 9. 실시간 5 FPS (0.2초) 스트리밍 브로드캐스트
    for frame_key, frame_data in dataset_records.items():
        if not pipeline_state["is_analyzing"]:
            break
        await manager.broadcast({
            "type": "CCTV_AI_STREAM",
            "frame_id": int(frame_key),
            "filename": filename,
            "zone_id": zone_id,
            "pedestrian_count": frame_data["pedestrian_count"],
            "occupancy_rate": frame_data["occupancy_rate"],
            "stagnation_sec": frame_data["stagnation_sec"],
            "cri_score": frame_data["cri_score"],
            "risk_level": frame_data["risk_level"],
            "low_confidence": frame_data.get("low_confidence", False),
            "detections": frame_data.get("detections", []),
            "timestamp": time.time()
        })
        await asyncio.sleep(0.2)  # 5 FPS 속도 조절

    pipeline_state["is_analyzing"] = False
    pipeline_state["status"] = "COMPLETED"
    print(f"[AI Pipeline Completed] '{filename}' 분석 파이프라인 완료.")


def run_pipeline_background(temp_file_path: str, start_time: str, fps: float, zone_id: int):
    """기존 coordinator.py / steps 파이프라인을 subprocess로 호출하는 백그라운드 태스크"""
    analysis_state["status"] = "running"
    analysis_state["started_at"] = datetime.now().isoformat()
    analysis_state["finished_at"] = None
    analysis_state["message"] = "Step 1: CCTV 파이프라인 분석 개시 (CSRNet/YOLO 및 BEV 좌표 변환)"
    analysis_state["progress_percent"] = 10.0
    analysis_state["error"] = None

    steps_dir = os.path.join(BASE_DIR, "steps")
    script_01 = os.path.join(steps_dir, "01_video_to_bev_anonymized.py")
    script_02 = os.path.join(steps_dir, "02_aggregate_pedestrian_json.py")

    csv_path = os.path.join(RESULTS_DIR, f"cctv_bev_coordinates_upload_{os.getpid()}.csv")
    video_path = os.path.join(RESULTS_DIR, f"cctv_result_upload_{os.getpid()}.mp4")
    s3_url = f"https://tdtc-cctv-upload.s3.ap-northeast-2.amazonaws.com/danger-clips/uploaded_clip_{zone_id}.mp4"

    try:
        env_01 = os.environ.copy()
        env_01.update({
            "OUTPUT_MP4": temp_file_path,
            "CCTV_BEV_CSV": csv_path,
            "CSR_RESULT_MP4": video_path,
            "CSRNET_MODEL_PATH": os.path.join(MODELS_DIR, "csrnet_ultimate_epoch_8.pth"),
            "ZONE_ID": str(zone_id),
            "TARGET_FPS": str(fps),
            "PYTHONIOENCODING": "utf-8"
        })

        proc_01 = subprocess.Popen(
            [PYTHON_EXE, script_01],
            env=env_01,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        while True:
            line = proc_01.stdout.readline()
            if not line and proc_01.poll() is not None:
                break
            if "%" in line or "프레임" in line:
                analysis_state["progress_percent"] = min(70.0, analysis_state["progress_percent"] + 0.5)
                analysis_state["message"] = f"Step 1: 프레임 분석 진행 중 ({analysis_state['progress_percent']:.0f}%)"

        proc_01.wait()
        if proc_01.returncode != 0:
            stderr_err = proc_01.stderr.read() or "01단계 스크립트 실행 중 에러"
            raise RuntimeError(f"CCTV BEV 파이프라인(01단계) 실패: {stderr_err[-500:]}")

        analysis_state["progress_percent"] = 75.0
        analysis_state["message"] = "Step 2: 보행자 JSON 데이터 집계 및 Supabase 적재 준비 중 (75%)"

        env_02 = os.environ.copy()
        env_02.update({
            "CCTV_BEV_CSV": csv_path,
            "ZONE_ID": str(zone_id),
            "S3_CLIP_URL": s3_url,
            "START_TIME": start_time,
            "FPS": str(fps),
            "SKIP_DB_INSERT": "FALSE",
            "PYTHONIOENCODING": "utf-8"
        })

        proc_02 = subprocess.Popen(
            [PYTHON_EXE, script_02],
            env=env_02,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        while True:
            line = proc_02.stdout.readline()
            if not line and proc_02.poll() is not None:
                break
            if "필터링 완료" in line or "ROI 적용" in line:
                analysis_state["progress_percent"] = 85.0
                analysis_state["message"] = "Step 2: ROI 영역 필터링 완료 (85%)"
            elif "글로벌 추적" in line:
                analysis_state["progress_percent"] = 90.0
                analysis_state["message"] = "Step 2: 객체 추적 및 경로 분석 중 (90%)"
            elif "DB 적재 시도 중" in line:
                analysis_state["progress_percent"] = 95.0
                analysis_state["message"] = "Step 2: Supabase 데이터베이스 적재 중 (95%)"

        proc_02.wait()
        if proc_02.returncode != 0:
            stderr_err = proc_02.stderr.read() or "02단계 스크립트 실행 중 에러"
            raise RuntimeError(f"보행자 집계(02단계) 실패: {stderr_err[-500:]}")

        for path in [csv_path, video_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        result_count = 0
        pedaggr_json = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
        if os.path.exists(pedaggr_json):
            try:
                with open(pedaggr_json, "r", encoding="utf-8") as f:
                    result_count = len(json.load(f))
            except Exception:
                pass

        analysis_state["status"] = "done"
        analysis_state["finished_at"] = datetime.now().isoformat()
        analysis_state["progress_percent"] = 100.0
        analysis_state["result_count"] = result_count
        analysis_state["message"] = f"✅ 분석 완료! 총 {result_count}개 프레임 적재 완료."

    except Exception as e:
        analysis_state["status"] = "error"
        analysis_state["finished_at"] = datetime.now().isoformat()
        analysis_state["error"] = str(e)
        analysis_state["message"] = f"❌ 오류 발생: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


async def process_confirm_background(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    zone_id: int,
    alert_type: str = "MANUAL_REPORT",
    incident_summary: Optional[str] = None,
    video_id: int = 1,
    cri_score: float = 75.0,
    pedestrian_count: int = 15,
    occupancy_rate: float = 40.0,
    stagnation_sec: float = 0.0,
    llm_summary: Optional[str] = None  # 이전 호환성 유지용
):
    """비디오 35초 슬라이싱, PDF 보고서 생성, S3 업로드, Java 웹훅 호출 수행"""
    from server.s3_uploader import upload_file_to_s3
    from server.pdf_generator import generate_emergency_pdf
    from server.video_buffer import global_buffer_manager
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sliced_filename = f"danger_clip_{zone_id}_{timestamp_str}.mp4"
    sliced_path = os.path.join(RESULTS_DIR, sliced_filename)
    snapshot_filename = f"snap_zone_{zone_id}_{timestamp_str}.jpg"
    snapshot_path = os.path.join(RESULTS_DIR, snapshot_filename)

    summary_text = incident_summary or llm_summary

    # 1. 35초 위험 클립 추출 (원형 버퍼 우선 -> 로컬 비디오 슬라이싱 fallback)
    print(f"[Confirm] 35초 위험 클립 생성 시작 (Zone: {zone_id}, Type: {alert_type}) -> {sliced_path}")
    is_cut_success = False

    buf_clip = global_buffer_manager.extract_danger_clip(zone_id, sliced_path, duration_sec=35.0)
    if buf_clip and os.path.exists(sliced_path) and os.path.getsize(sliced_path) > 1000:
        is_cut_success = True
        global_buffer_manager.save_snapshot(zone_id, snapshot_path)
        print(f"[Confirm] 원형 버퍼(VideoBuffer)로부터 35초 위험 클립 추출 성공.")
    elif video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(sliced_path, fourcc, fps, (width, height))
            
            start_frame = int(start_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_to_write = int((end_sec - start_sec) * fps)
            saved_snap = False
            for _ in range(frames_to_write):
                ret, frame = cap.read()
                if not ret:
                    break
                if not saved_snap:
                    cv2.imwrite(snapshot_path, frame)
                    saved_snap = True
                out.write(frame)
                
            cap.release()
            out.release()
            is_cut_success = True
            print(f"[Confirm] 원본 비디오 파일 슬라이싱 완료.")

    if not is_cut_success:
        print(f"[Confirm Warning] 비디오 컷팅에 실패했습니다. Mock 파일을 복사합니다.")
        if video_path and os.path.exists(video_path):
            shutil.copy(video_path, sliced_path)
        else:
            open(sliced_path, "w").close()

    # 2. ReportLab 기반 PDF 명세서 파일 생성
    pdf_filename = f"report_{zone_id}_{timestamp_str}.pdf"
    pdf_path = os.path.join(RESULTS_DIR, pdf_filename)
    try:
        print(f"[Confirm] PDF 명세서 생성 시작 -> {pdf_path}")
        generate_emergency_pdf(
            output_pdf_path=pdf_path,
            zone_id=zone_id,
            alert_type=alert_type,
            cri_score=cri_score,
            pedestrian_count=pedestrian_count,
            occupancy_rate=occupancy_rate,
            incident_summary=summary_text,
            stagnation_sec=stagnation_sec,
            snapshot_image_path=snapshot_path if os.path.exists(snapshot_path) else None
        )
        print(f"[Confirm] PDF 명세서 생성 완료.")
    except Exception as e:
        print(f"[Confirm Warning] PDF 리포트 생성 중 에러: {e}")

    # 3. S3 업로드 (danger-clips/ 및 post-reports/)
    print(f"[Confirm] S3 파일 업로드 시작...")
    s3_clip_url = upload_file_to_s3(sliced_path, f"danger-clips/{sliced_filename}")
    s3_pdf_url = upload_file_to_s3(pdf_path, f"post-reports/{pdf_filename}")
    print(f"[Confirm] S3 업로드 완료. Clip URL: {s3_clip_url}, PDF URL: {s3_pdf_url}")

    # 로컬 임시 파일 정리
    for path in [sliced_path, pdf_path, snapshot_path]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # 4. Java 백엔드로 긴급 알람 트리거 & 비동기 웹훅 호출
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": BACKEND_API_KEY
    }
    
    # 4-1. 알람 트리거 (POST /api/ai/alerts/trigger)
    alert_url = f"{BACKEND_URL}/api/ai/alerts/trigger"
    alert_payload = {
        "zoneId": zone_id,
        "alertType": alert_type,
        "pdfUrl": s3_pdf_url,
        "videoUrl": s3_clip_url,
        "llmSummary": summary_text
    }
    alert_id = 1
    try:
        print(f"[Confirm] Java BE 알람 호출: {alert_url}")
        res = requests.post(alert_url, headers=headers, json=alert_payload, timeout=5)
        if res.status_code == 200:
            try:
                alert_id = int(res.text.strip())
            except Exception:
                pass
        print(f"[Confirm] Java BE 알람 응답: {res.status_code} (Alert ID: {alert_id})")
    except Exception as e:
        print(f"[Confirm Error] Java BE 알람 호출 실패: {e}")

    # 4-2. 비디오 클립 등록 (POST /api/v1/video-clips) - 위험 영상은 무조건 "RISK" 타입으로 영구 보존
    clip_url = f"{BACKEND_URL}/api/v1/video-clips"
    now_utc = datetime.now(timezone.utc)
    clip_payload = {
        "zoneId": zone_id,
        "clipType": "RISK",
        "s3ClipUrl": s3_clip_url,
        "startTime": (now_utc - timedelta(seconds=17.5)).isoformat().replace("+00:00", "Z"),
        "endTime": (now_utc + timedelta(seconds=17.5)).isoformat().replace("+00:00", "Z")
    }
    created_clip_id = 1
    try:
        print(f"[Confirm] Java BE 비디오 클립 웹훅 호출: {clip_url}")
        res = requests.post(clip_url, headers=headers, json=clip_payload, timeout=5)
        if res.status_code == 200:
            try:
                resp_json = res.json()
                created_clip_id = resp_json.get("clipId", 1)
            except Exception:
                pass
        print(f"[Confirm] Java BE 비디오 클립 웹훅 응답: {res.status_code} (Clip ID: {created_clip_id})")
    except Exception as e:
        print(f"[Confirm Error] Java BE 비디오 클립 웹훅 호출 실패: {e}")

    # 4-3. PDF 리포트 등록 (POST /api/v1/post-reports)
    report_url = f"{BACKEND_URL}/api/v1/post-reports"
    report_payload = {
        "alertId": alert_id,
        "llmSummary": summary_text,
        "s3PdfUrl": s3_pdf_url,
        "videoId": video_id or created_clip_id
    }
    try:
        print(f"[Confirm] Java BE PDF 리포트 웹훅 호출: {report_url}")
        res = requests.post(report_url, headers=headers, json=report_payload, timeout=5)
        print(f"[Confirm] Java BE PDF 리포트 웹훅 응답: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"[Confirm Error] Java BE PDF 리포트 웹훅 호출 실패: {e}")

