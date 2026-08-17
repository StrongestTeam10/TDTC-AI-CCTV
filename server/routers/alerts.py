"""
server/routers/alerts.py - 긴급 알람 트리거 및 최근 결과 조회 라우터
"""

import os
import json
import requests
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from server.config import (
    BACKEND_URL, BACKEND_API_KEY, RESULTS_DIR, AlertTriggerRequest
)

router = APIRouter(tags=["알람"])


@router.post("/api/alerts/trigger")
def trigger_alert(request: AlertTriggerRequest):
    """Java 백엔드 /api/ai/alerts/trigger를 호출하여 긴급 알람 발생"""
    url = f"{BACKEND_URL}/api/ai/alerts/trigger"
    headers = {"Content-Type": "application/json", "X-API-KEY": BACKEND_API_KEY}
    payload = {"zoneId": request.zone_id, "alertType": request.alert_type}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return {"message": "✅ 긴급 알람이 Java 백엔드로 전송됐습니다.", "backend_response": response.text}
        raise HTTPException(status_code=response.status_code, detail=f"Java 백엔드 오류: {response.text}")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Java 백엔드({BACKEND_URL})에 연결할 수 없습니다.")


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
