# cleanup_cctv_ai_pipeline.py
import os
import shutil

CCTV_AI_DIR = r"e:\AIVLE_10team\ai_pipeline\cctv_ai_pipeline"
ARCHIVE_DIR = os.path.join(CCTV_AI_DIR, "legacy_archive", "old_scripts")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# cctv_ai_pipeline 내의 최상위 파일들을 legacy_archive/old_scripts 로 이동
for item in os.listdir(CCTV_AI_DIR):
    item_path = os.path.join(CCTV_AI_DIR, item)
    if os.path.isfile(item_path):
        if item.lower() in [".env", "readme.md"]:
            continue
        try:
            shutil.move(item_path, os.path.join(ARCHIVE_DIR, item))
            print(f"[ARCHIVED] {item} -> legacy_archive/old_scripts/{item}")
        except Exception as e:
            print(f"[SKIP] {item}: {e}")

# 안내용 README.md 작성/업데이트
readme_path = os.path.join(CCTV_AI_DIR, "README.md")
readme_content = """# ⚠️ CCTV 파이프라인 모듈 이관 안내 (Deprecated Directory)

본 디렉토리(`cctv_ai_pipeline`)의 모든 구형 파이프라인 모듈 및 스크립트는 **신규 `cctv_upload/` 파이프라인 구조로 100% 통합 이관 및 리팩토링**되었습니다.

* **신규 파이프라인 위치**: [`ai_pipeline/cctv_upload/`](file:///e:/AIVLE_10team/ai_pipeline/cctv_upload)
  * **핵심 연산 코어**: `cctv_upload/core/` ([coordinator.py](file:///e:/AIVLE_10team/ai_pipeline/cctv_upload/core/coordinator.py), [batch_pipeline.py](file:///e:/AIVLE_10team/ai_pipeline/cctv_upload/core/batch_pipeline.py))
  * **비식별화/대시보드**: `cctv_upload/monitors/` ([api_server.py](file:///e:/AIVLE_10team/ai_pipeline/cctv_upload/monitors/api_server.py))
  * **단계별 스크립트**: `cctv_upload/steps/`
  * **기술 문서**: `cctv_upload/docs/`

기존 구형 스크립트들은 `legacy_archive/old_scripts/` 디렉토리로 보관 처리되었습니다.
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ cctv_ai_pipeline 폴더 정리 및 이관 안내 README 작성 완료!")
