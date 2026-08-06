# 🎥 CCTV AI Pipeline Directory

망원시장 CCTV 기반 보행자 감지, 3D BEV 공간 변환, 다차원 위험도 스코어링 및 Supabase 백엔드 적재 파이프라인 폴더입니다.

---

## 📌 문서 바로가기
* 📖 **[PIPELINE.md 바로가기](file:///e:/AIVLE_10team/cctv_ai_pipeline/PIPELINE.md)**: CCTV AI 파이프라인의 01~10번 단계별 스크립트 역할, 입출력 파일 및 구조 가이드
* 📄 **[BACKEND.md 바로가기](file:///e:/AIVLE_10team/BACKEND.md)**: Supabase 백엔드 데이터베이스 적재 순서, 테이블 스키마 및 컬럼 명세서

---

## 🚀 빠른 실행

```bash
# 통합 파이프라인 실행
python cctv_ai_pipeline/00_run_cctv_only_pipeline.py

# 보행자 JSON 집계 및 DB 적재 파이프라인 실행
python cctv_ai_pipeline/09_aggregate_pedestrian_json.py
```
