"""
ai_server.py - CCTV AI 분석 통합 FastAPI 서버 (Refactored)
=============================================
FastAPI 앱 정의 및 모듈화된 라우터 통합 진입점

실행 방법:
    uvicorn ai_server:app --host 0.0.0.0 --port 8088 --reload
"""

import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import (
    BASE_DIR, BACKEND_URL, MODELS_DIR, RESULTS_DIR, pipeline_state, analysis_state
)
from server.models import torch_available
from server.websocket import manager
from server.routers import cctv, analyze, alerts

# =========================================================================
# FastAPI 앱 수명주기(Lifespan) 관리
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 TDTC CCTV AI 통합 FastAPI 서버 가동! (Refactored)")
    print(f"   백엔드 URL   : {BACKEND_URL}")
    print(f"   모델 경로    : {MODELS_DIR}")
    print(f"   결과 저장    : {RESULTS_DIR}")
    print(f"   PyTorch      : {'✅ 사용 가능' if torch_available else '❌ 미설치'}")
    print("=" * 60)
    yield
    print("🛑 서버 종료")


app = FastAPI(
    title="TDTC CCTV AI 통합 분석 서버",
    description=(
        "CSRNet 기반 CCTV 보행자 분석 파이프라인 API\n\n"
        "- **WebSocket 스트리밍**: `/ws/cctv-stream`\n"
        "- **외부 접속**: https://tdtc-ai-cctv.uk\n"
        "- **Swagger UI**: https://tdtc-ai-cctv.uk/docs"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 공통 헬스체크 및 엠블럼 엔드포인트
# =========================================================================
@app.get("/", tags=["공통"])
async def root():
    """서버 상태 및 연결 정보"""
    return {
        "status": "ONLINE",
        "service": "TDTC CCTV AI 통합 분석 서버",
        "version": "2.0.0",
        "active_ws_clients": len(manager.active_connections),
        "pipeline_state": pipeline_state,
        "analysis_state": analysis_state["status"],
        "swagger_ui": "https://tdtc-ai-cctv.uk/docs",
    }


@app.get("/health", tags=["공통"])
def health_check():
    """서버 및 GPU 상태 확인"""
    gpu_available = False
    gpu_name = "없음"
    if torch_available:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "없음"
        except Exception:
            pass
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {"available": gpu_available, "device_name": gpu_name},
        "torch_available": torch_available,
        "subprocess_pipeline_status": analysis_state["status"],
        "websocket_pipeline_status": pipeline_state["status"],
    }


# =========================================================================
# 라우터 등록
# =========================================================================
app.include_router(cctv.router)
app.include_router(analyze.router)
app.include_router(alerts.router)


# =========================================================================
# 직접 실행 시
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8088, reload=True)
