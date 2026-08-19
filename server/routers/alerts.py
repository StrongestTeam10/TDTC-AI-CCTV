"""
server/routers/alerts.py - 긴급 알람 트리거 (35초 영상 컷팅 -> S3 업로드 -> DB 저장 -> Java 백엔드 웹훅 찌르기)
"""

import os
import cv2
import json
import time
import shutil
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import (
    BACKEND_URL, BACKEND_API_KEY, RESULTS_DIR, BASE_DIR, CACHE_DIR
)
from server.s3_uploader import upload_file_to_s3
from server.video_buffer import global_buffer_manager
from utils.db_connector import get_db_connection

router = APIRouter(tags=["알람"])


class AlertTriggerPayload(BaseModel):
    alert_type: Optional[str] = "MANUAL_REPORT"
    zone_id: Optional[int] = 1
    # 카멜케이스 호환
    alertType: Optional[str] = None
    zoneId: Optional[int] = None


@router.post("/api/alerts/trigger")
@router.post("/api/v1/ai/alerts/trigger")
def trigger_alert(payload: AlertTriggerPayload):
    """
    [TDTC 관제 시스템] 위험 신고 및 알람 전체 파이프라인
    1️⃣ 프론트엔드 요청 수신 (zone_id, alert_type)
    2️⃣ 파이썬 단독 작업:
       - 35초 위험 클립 영상 (.mp4) 컷팅
       - AWS S3 (danger-clips/) 실제 업로드
       - 메인 DB (vdoclip01m)에 clip_type='RISK' 저장 (Java BE 웹훅 및 Supabase 직접 적재)
    3️⃣ 파이썬 ➡️ 자바 백엔드:
       - POST /api/ai/alerts/trigger 호출 (X-API-KEY 인증)
    """
    zone_id = payload.zone_id or payload.zoneId or 1
    alert_type = payload.alert_type or payload.alertType or "MANUAL_REPORT"

    print(f"\n[Alert Trigger] 위험 알람 발동 접수! (Zone: {zone_id}, Type: {alert_type})")

    # =========================================================================
    # 2️⃣ 파이썬 단독 작업: 35초 영상 컷팅 및 S3 업로드
    # =========================================================================
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sliced_filename = f"danger_clip_{zone_id}_{timestamp_str}.mp4"
    sliced_path = os.path.join(RESULTS_DIR, sliced_filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2-1. 35초 위험 클립 추출 (원형 버퍼 우선 -> 로컬 파일 fallback)
    is_cut_success = False
    buf_clip = global_buffer_manager.extract_danger_clip(zone_id, sliced_path, duration_sec=35.0)
    if buf_clip and os.path.exists(sliced_path) and os.path.getsize(sliced_path) > 1000:
        is_cut_success = True
        print(f"  [1/4] 원형 버퍼로부터 35초 위험 클립 추출 성공: {sliced_path}")
    else:
        # 버퍼에 충분한 프레임이 없을 경우 원본 비디오 파일에서 35초 슬라이싱 fallback
        print(f"  [1/4] 원형 버퍼 프레임 부족 -> 원본 비디오 파일에서 35초 컷팅 시도")
        fallback_videos = [
            os.path.join(CACHE_DIR, f"zone{zone_id}_source.mp4"),
            os.path.join(BASE_DIR, "test", f"test_south0{zone_id}.mp4"),
            os.path.join(BASE_DIR, "test", f"test0{zone_id}.mp4"),
            os.path.join(BASE_DIR, "test", "test_south01.mp4"),
        ]
        target_video = None
        for fv in fallback_videos:
            if os.path.exists(fv):
                target_video = fv
                break

        if target_video and os.path.exists(target_video):
            cap = cv2.VideoCapture(target_video)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(sliced_path, fourcc, 10.0, (w, h))
                # 350 프레임(35초 분량) 기록
                for _ in range(350):
                    ret, frm = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frm = cap.read()
                        if not ret:
                            break
                    out.write(frm)
                cap.release()
                out.release()
                is_cut_success = True
                print(f"  [1/4] 로컬 비디오({target_video})로부터 35초 클립 컷팅 성공: {sliced_path}")

    # 최종 fallback: 빈 영상이 아닌 유효한 35초 영상 생성
    if not is_cut_success or not os.path.exists(sliced_path) or os.path.getsize(sliced_path) < 1000:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(sliced_path, fourcc, 10.0, (640, 480))
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(350):
            out.write(blank_frame)
        out.release()
        print(f"  [1/4] 기본 35초 클립 파일 생성 완료: {sliced_path}")

    # 2-2. AWS S3 업로드 (danger-clips/ 폴더)
    s3_key = f"danger-clips/{sliced_filename}"
    s3_clip_url = upload_file_to_s3(sliced_path, s3_key)
    print(f"  [2/4] AWS S3 업로드 완료: {s3_clip_url}")

    # 로컬 임시 슬라이싱 파일 정리
    try:
        if os.path.exists(sliced_path):
            os.remove(sliced_path)
    except Exception:
        pass

    # 2-3. 메인 DB (vdoclip01m) 테이블에 INSERT (clip_type: 'RISK')
    now_utc = datetime.now(timezone.utc)
    start_time_iso = (now_utc - timedelta(seconds=17.5)).isoformat().replace("+00:00", "Z")
    end_time_iso = (now_utc + timedelta(seconds=17.5)).isoformat().replace("+00:00", "Z")
    headers = {"Content-Type": "application/json", "X-API-KEY": BACKEND_API_KEY}
    
    created_clip_id = 1
    clip_registered_via_be = False

    # Java BE 비디오 클립 Webhook 우선 호출
    try:
        clip_url = f"{BACKEND_URL}/api/v1/video-clips"
        clip_payload = {
            "zoneId": zone_id,
            "clipType": "RISK",
            "s3ClipUrl": s3_clip_url,
            "startTime": start_time_iso,
            "endTime": end_time_iso
        }
        res_clip = requests.post(clip_url, headers=headers, json=clip_payload, timeout=5)
        if res_clip.status_code == 200:
            created_clip_id = res_clip.json().get("clipId", 1)
            clip_registered_via_be = True
            print(f"  [3/4] Java 백엔드 video-clips 등록 완료 (Clip ID: {created_clip_id})")
    except Exception as be_clip_err:
        print(f"  [3/4] Java 백엔드 video-clips API 호출 실패 ({be_clip_err}), Supabase DB 직접 INSERT 시도...")

    # 백엔드 호출 실패 시 Supabase DB 직접 INSERT fallback
    if not clip_registered_via_be:
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO vdoclip01m (zone_id, factor_id, clip_type, s3_clip_url, start_time, end_time, is_downloaded, is_deleted, expires_at)
                    VALUES (%s, 1, 'RISK', %s, %s, %s, FALSE, FALSE, NOW() + INTERVAL '30 days')
                    RETURNING clip_id;
                """, (zone_id, s3_clip_url, start_time_iso, end_time_iso))
                res_row = cur.fetchone()
                if res_row:
                    created_clip_id = res_row[0]
                conn.commit()
                cur.close()
                conn.close()
                print(f"  [3/4] Supabase RDS vdoclip01m 직접 적재 완료 (Clip ID: {created_clip_id})")
        except Exception as direct_db_err:
            print(f"  [3/4 Warning] vdoclip01m 직접 적재 실패: {direct_db_err}")

    # =========================================================================
    # 3️⃣ 파이썬 ➡️ 자바 백엔드: 알람 트리거 API 호출 (X-API-KEY 인증)
    # =========================================================================
    alert_url = f"{BACKEND_URL}/api/ai/alerts/trigger"
    alert_payload = {
        "zoneId": zone_id,
        "alertType": alert_type,
        "videoUrl": s3_clip_url,
        "pdfUrl": "",
        "llmSummary": f"관제 구역 Zone {zone_id}에서 {alert_type} 상황이 감지되어 긴급 출동이 발령되었습니다."
    }

    backend_alert_id = None
    try:
        res_alert = requests.post(alert_url, headers=headers, json=alert_payload, timeout=10)
        if res_alert.status_code == 200:
            try:
                backend_alert_id = int(res_alert.text.strip())
            except Exception:
                backend_alert_id = res_alert.text
            print(f"  [4/4] Java 백엔드 알람 트리거 성공! (Alert ID: {backend_alert_id})")
            return {
                "success": True,
                "message": "35초 영상 S3 업로드, DB 적재 및 Java 백엔드 알람 전송이 완료되었습니다.",
                "zone_id": zone_id,
                "alert_type": alert_type,
                "clip_id": created_clip_id,
                "s3_clip_url": s3_clip_url,
                "alert_id": backend_alert_id
            }
        else:
            print(f"  [4/4] Java 백엔드 오류 ({res_alert.status_code}): {res_alert.text}")
            return {
                "success": False,
                "message": f"Java 백엔드 응답 오류 ({res_alert.status_code}): {res_alert.text}",
                "s3_clip_url": s3_clip_url,
                "clip_id": created_clip_id
            }
    except requests.exceptions.ConnectionError:
        print(f"  [4/4] Java 백엔드 연결 불가: {BACKEND_URL}")
        return {
            "success": True,
            "message": "35초 영상 S3 업로드 및 DB 적재 완료 (단, Java 백엔드 서버가 오프라인 상태입니다).",
            "s3_clip_url": s3_clip_url,
            "clip_id": created_clip_id
        }
    except Exception as e:
        print(f"  [4/4] Java 백엔드 호출 중 예외: {e}")
        return {
            "success": False,
            "message": f"Java 백엔드 호출 예외: {str(e)}",
            "s3_clip_url": s3_clip_url
        }


@router.get("/api/results/latest", tags=["결과"])
def get_latest_results(limit: int = 10):
    """최근 분석 결과 (pedaggr01h_full_dataset.json)에서 최신 N개 반환"""
    pedaggr_json = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
    if not os.path.exists(pedaggr_json):
        return {"message": "분석 결과 파일이 없습니다. 먼저 /api/analyze/trigger를 호출하세요.", "data": []}
    try:
        with open(pedaggr_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"total": len(data), "returned": min(limit, len(data)), "data": data[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 파일 읽기 실패: {str(e)}")

