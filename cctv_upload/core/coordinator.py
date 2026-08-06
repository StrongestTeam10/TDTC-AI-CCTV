# load_zone_pedestrian_data.py
import os
import sys
import json
import time
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
CCTV_UPLOAD_DIR = os.path.dirname(CORE_DIR)
AI_PIPELINE_DIR = os.path.dirname(CCTV_UPLOAD_DIR)
WORKSPACE_DIR = os.path.dirname(AI_PIPELINE_DIR)

STEPS_DIR = os.path.join(CCTV_UPLOAD_DIR, "steps")
if not os.path.exists(STEPS_DIR):
    STEPS_DIR = os.path.join(AI_PIPELINE_DIR, "cctv_ai_pipeline")

TEST_DIR = os.path.join(AI_PIPELINE_DIR, "test")
if not os.path.exists(TEST_DIR):
    TEST_DIR = os.path.join(WORKSPACE_DIR, "test")

RESULTS_DIR = os.path.join(AI_PIPELINE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 실행 인터프리터 설정
PYTHON_EXE = sys.executable


def scan_video_files():
    """test/zone_id X 폴더 내의 비디오 파일들을 수집 및 정렬"""
    zones = {1: [], 2: [], 3: []}
    for zone in [1, 2, 3]:
        zone_path = os.path.join(TEST_DIR, f"zone_id {zone}")
        if not os.path.exists(zone_path):
            print(f"[WARNING] 구역 폴더가 없습니다: {zone_path}")
            continue
        
        files = [f for f in os.listdir(zone_path) if f.lower().endswith(".mp4")]
        files.sort()  # 파일명 순 정렬
        zones[zone] = [os.path.join(zone_path, f) for f in files]
    
    return zones

def run_step(script_name, env_vars):
    """지정한 환경변수를 주입하여 서브프로세스로 스크립트 실행"""
    script_path = os.path.join(STEPS_DIR, script_name)
    if not os.path.exists(script_path):
        script_path = os.path.join(CORE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[ERROR] 스크립트가 없습니다: {script_path}")
        return False
    
    env = os.environ.copy()
    env.update(env_vars)
    env["PYTHONIOENCODING"] = "utf-8"
    
    print(f"[SUBPROCESS] {script_name} 실행 중... (입력 비디오: {env_vars.get('OUTPUT_MP4')})")
    res = subprocess.run([PYTHON_EXE, script_path], env=env)
    return res.returncode == 0


def main():
    start_wall_time = time.time()
    parser = argparse.ArgumentParser(description="3개 구역 비디오 데이터 실시간 순서 교차 적재 툴")
    parser.add_argument("--test-run", action="store_true", help="각 구역별 첫 번째 비디오만 테스트용으로 분석 실행")
    parser.add_argument("--start-time", default="2026-08-05 14:00:00", help="비디오의 시작 기준 시간 (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--fps", type=float, default=10.0, help="영상 분석 시 적용할 FPS")
    parser.add_argument("--legacy-mode", action="store_true", help="기존 직렬/분산 방식 사용 여부")
    args = parser.parse_args()
    args.realtime_batch = not args.legacy_mode



    print("="*80)
    print(" [CCTV 3개 구역 실시간 교차 적재 코디네이터] ")
    print(f" Workspace: {WORKSPACE_DIR}")
    print("="*80)

    # 1. 비디오 파일 스캔
    video_map = scan_video_files()
    max_groups = max(len(video_map[1]), len(video_map[2]), len(video_map[3]))
    
    if max_groups == 0:
        print("[ERROR] 처리할 비디오 파일(.mp4)을 찾을 수 없습니다. test/ 폴더를 확인하세요.")
        sys.exit(1)
        
    print(f"[INFO] 스캔 완료 - 남쪽(Zone 1): {len(video_map[1])}개, 중앙(Zone 2): {len(video_map[2])}개, 북쪽(Zone 3): {len(video_map[3])}개")

    # 2. 비디오 그룹화 (동시 시점의 영상들을 하나씩 묶음)
    groups_to_process = []
    limit = 1 if args.test_run else max_groups
    
    for i in range(limit):
        group = {}
        for zone in [1, 2, 3]:
            if i < len(video_map[zone]):
                group[zone] = video_map[zone][i]
        groups_to_process.append(group)

    print(f"[INFO] 총 {len(groups_to_process)}개의 비디오 그룹을 분석합니다." + (" (테스트 실행 모드)" if args.test_run else ""))

    # 시작 기준 시각 파싱
    try:
        base_start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"[ERROR] 시간 형식이 올바르지 않습니다: {args.start_time} (올바른 형식: YYYY-MM-DD HH:MM:SS)")
        sys.exit(1)

    all_pedaggr_records = []

    # 3. 비디오 그룹별 순회 분석
    for idx, group in enumerate(groups_to_process):
        print(f"\n[그룹 {idx+1}/{len(groups_to_process)}] 영상 분석 시작 (기준 시각: {base_start_time})")

        # ===== 구역별 메타 사전 미리 구성 =====
        zone_meta = {}
        for zone, video_path in group.items():
            clip_id = 100 + (idx * 3) + zone
            temp_csv_path = os.path.join(RESULTS_DIR, f"temp_bev_clip_{clip_id}.csv")
            s3_url = f"https://mangwon-cctv.s3.ap-northeast-2.amazonaws.com/danger-clips/clip_{clip_id}.mp4"
            zone_meta[zone] = {
                "video_path": video_path,
                "clip_id": clip_id,
                "temp_csv_path": temp_csv_path,
                "s3_url": s3_url
            }

        # ===== Realtime Batch Engine 사용 조건 =====
        if args.realtime_batch:
            print(f"\n[=== Realtime Batch Engine: 3개 구역 통합 배치 분석 시작 ===]")
            try:
                try:
                    from core.batch_pipeline import RealtimeBatchPipeline
                except ImportError:
                    from batch_pipeline import RealtimeBatchPipeline

                batch_pipeline = RealtimeBatchPipeline(group, target_fps=args.fps)

                batch_results = batch_pipeline.run()

                # 결과를 pedaggr 레코드 형식으로 패킹
                for zone_id, records in batch_results.items():
                    clip_id = 100 + (idx * 3) + zone_id
                    s3_url = f"https://mangwon-cctv.s3.ap-northeast-2.amazonaws.com/danger-clips/clip_{clip_id}.mp4"
                    
                    for r in records:
                        # 프레임 타임스탬프 계산
                        frame_time = base_start_time + timedelta(seconds=r["timestamp"])
                        
                        # 보행자 ID를 매핑한 JSON 오브젝트 구성 (new_table.txt 스펙 매칭)
                        pixels_map = {}
                        bev_xyz_map = {}
                        for i, c in enumerate(r["coordinates"]):
                            pid = f"person_{i+1}"
                            pixels_map[pid] = {"x": round(float(c["px"]), 2), "y": round(float(c["py"]), 2)}
                            bev_xyz_map[pid] = {"x": round(float(c["bev_x"]), 3), "y": round(float(c["bev_y"]), 3), "z": 0.0}
                        
                        rec = {
                            "frame_id": r["frame_idx"],
                            "pixels_json": json.dumps(pixels_map, ensure_ascii=False),
                            "bev_xyz_json": json.dumps(bev_xyz_map, ensure_ascii=False),
                            "captured_at": frame_time.isoformat(),
                            "clip_id": clip_id,
                            "video_id": 1,
                            "zone_id": zone_id,
                            "s3_clip_url": s3_url
                        }
                        all_pedaggr_records.append(rec)
                
                print(f"[SUCCESS] 그룹 {idx+1} Realtime Batch 분석 완료!")
                base_start_time += timedelta(minutes=1)
                continue
            except Exception as e:
                print(f"[ERROR] Realtime Batch Pipeline 실행 도중 예외 발생: {e}, 기존 방식으로 fallback 합니다.")

        # ===== Phase 1: 3개 구역 04단계 (CSRNet BEV 추출)를 동시 병렬 실행 =====

        def run_zone_04(zone):
            meta = zone_meta[zone]
            v_path = meta["video_path"]
            csv_path = meta["temp_csv_path"]

            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10:
                print(f"[INFO] Zone {zone} 임시 BEV CSV 이미 존재, 04단계 스킵: {csv_path}")
                return zone, True, 0.0

            env_04 = {
                "OUTPUT_MP4": v_path,
                "CCTV_BEV_CSV": csv_path,
                "CSRNET_MODEL_PATH": os.path.join(WORKSPACE_DIR, "models", "csrnet_ultimate_epoch_8.pth"),
                "DATASET_TYPE": "MALL",
                "ZONE_ID": str(zone),  # 각 구역의 ZONE_ID 공급
                "TARGET_FPS": str(args.fps)  # 타겟 FPS 주입 (15 하드코딩 버그 해결)
            }
            script_path = os.path.join(STEPS_DIR, "04_video_to_bev_CSR.py")
            if not os.path.exists(script_path):
                script_path = os.path.join(CORE_DIR, "04_video_to_bev_CSR.py")

            env = os.environ.copy()
            env.update(env_04)
            env["PYTHONIOENCODING"] = "utf-8"
            t0 = time.time()
            res = subprocess.run([PYTHON_EXE, script_path], env=env)
            elapsed = time.time() - t0
            return zone, res.returncode == 0, elapsed

        print(f"\n[=== Phase 1: 3개 구역 BEV 추출 병렬 실행 시작 ===]")
        phase1_start = time.time()
        zone_04_ok = {}

        with ThreadPoolExecutor(max_workers=min(len(group), 2)) as executor:  # GPU OOM 방지: 최대 2개 동시 실행
            futures = {executor.submit(run_zone_04, z): z for z in group.keys()}
            for future in as_completed(futures):
                zone, success, elapsed = future.result()
                zone_04_ok[zone] = success
                if success and elapsed > 0:
                    print(f"[소요 시간] Zone {zone} BEV 추출 완료: {elapsed:.2f}초")
                elif not success:
                    print(f"[ERROR] Zone {zone} BEV 추출 실패!")

        print(f"[=== Phase 1 완료] 전체 병렬 실행 시간: {time.time() - phase1_start:.2f}초 ===]")

        # ===== Phase 2: 각 구역 09단계 (집계) 순차 실행 (출력 파일 충돌 방지) =====
        print(f"\n[=== Phase 2: 3개 구역 보행자 집계 순차 실행 ===]")
        for zone in sorted(group.keys()):
            if not zone_04_ok.get(zone, False):
                print(f"[SKIP] Zone {zone}: 04단계 실패로 건너뀤니다.")
                continue

            meta = zone_meta[zone]
            clip_id = meta["clip_id"]
            temp_csv_path = meta["temp_csv_path"]
            s3_url = meta["s3_url"]

            env_09 = {
                "INPUT_PEDESTRIAN_CSV": temp_csv_path,
                "CLIP_ID": str(clip_id),
                "ZONE_ID": str(zone),
                "S3_CLIP_URL": s3_url,
                "START_TIME": base_start_time.isoformat(),
                "FPS": str(args.fps),
                "SKIP_DB_INSERT": "TRUE"
            }
            success = run_step("09_aggregate_pedestrian_json.py", env_09)
            if not success:
                print(f"[ERROR] Zone {zone} 보행자 JSON 집계 실패!")
                continue

            temp_json_path = os.path.join(RESULTS_DIR, "pedaggr01h_full_dataset.json")
            if os.path.exists(temp_json_path):
                try:
                    with open(temp_json_path, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        all_pedaggr_records.extend(records)
                        print(f"[SUCCESS] Zone {zone} 데이터 수집 완료: {len(records)} 프레임")
                except Exception as e:
                    print(f"[ERROR] 임시 JSON 파싱 실패: {e}")

            if os.path.exists(temp_csv_path):
                try:
                    os.remove(temp_csv_path)
                except Exception:
                    pass

        base_start_time += timedelta(minutes=1)

    print(f"\n" + "="*80)
    print(f"[분석 완료] 수집된 전체 레코드 수: {len(all_pedaggr_records)}개")
    print("="*80)

    if not all_pedaggr_records:
        print("[WARNING] 적재할 보행자 데이터가 존재하지 않습니다.")
        return

    # 4. 전체 레코드 시간(captured_at) 기준으로 교차 정렬
    print("[INFO] captured_at 컬럼을 기준으로 전체 레코드 정렬 중...")
    # ISO8601 문자열 기준 정렬도 연/월/일 시:분:초 형태이므로 문자열 정렬로 가능
    all_pedaggr_records.sort(key=lambda x: x.get("captured_at", ""))
    print("[SUCCESS] 정렬 완료!")

    # 5. DB 일괄 적재 실행
    sys.path.append(CORE_DIR)
    sys.path.append(CCTV_UPLOAD_DIR)
    sys.path.append(AI_PIPELINE_DIR)
    sys.path.append(os.path.join(AI_PIPELINE_DIR, "cctv_ai_pipeline", "sensor_fusion_archive"))
    
    try:
        # pyrefly: ignore [missing-import]
        from utils.db_connector import bulk_insert_pedestrian_coordinate_json
        print(f"\n[DB 적재] Supabase로 {len(all_pedaggr_records)}개 레코드 일괄 적재 시작...")
        
        # 날씨 가중치 정보 수집
        weather_info = None
        try:
            try:
                from core.weather_api import get_mangwon_weather
            except ImportError:
                from weather_api import get_mangwon_weather
            weather_info = get_mangwon_weather(fallback_mode="SUNNY")
        except Exception:
            pass

            
        db_success = bulk_insert_pedestrian_coordinate_json(all_pedaggr_records, weather_info=weather_info)
        if db_success:
            print(f"[SUCCESS] 3개 구역 교차 시간순 적재 완료!")
        else:
            print(f"[FAILURE] Supabase DB 적재 도중 오류가 발생했습니다.")
            
    except Exception as e:
        print(f"[ERROR] DB 적재 실패: {e}")

    print(f"\n" + "="*80)
    print(f"[총 소요 시간] 전체 프로세스 완료: {time.time() - start_wall_time:.2f}초 소요")
    print("="*80)

if __name__ == "__main__":
    main()
