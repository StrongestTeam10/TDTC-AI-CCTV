# run_cctv_only_pipeline.py
# CCTV 프레임 처리, CSRNet/YOLO 기반 인원 감지, BEV 바닥 좌표계 역투영(2D->3D)까지
# CCTV 파이프라인만 독립적으로 실행하고 관리하는 메인 통합 스크립트입니다.
import os
import sys
import subprocess
import time

# =========================================================================
# [설정] 통합 경로 및 파라미터 정의 (경로 변경 시 여기만 수정하세요)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 실행할 파이썬 가상환경 인터프리터 경로
PYTHON_EXE = os.environ.get("PIPELINE_PYTHON", r"E:\anaconda\envs\lidar_env\python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

# 1) 입력 데이터 경로
IMAGE_DIR = r"E:\test\archive\frames\frames"                # CCTV 프레임 이미지 폴더 (Mall Dataset 이미지 경로)
LABEL_DIR = r"E:\test\archive"                             # GT 정답 라벨 폴더 (Mall Dataset)

# 2) 출력 및 결과물 저장 경로
OUTPUT_DIR = r"E:\test"                          # 변환된 비디오(.mp4)가 저장될 폴더
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "cctv_mall_output.mp4")
CSR_RESULT_MP4 = os.path.join(OUTPUT_DIR, "cctv_mall_result.mp4")

# 3) 결과 CSV 및 모델 가중치 경로
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CCTV_BEV_CSV = os.path.join(RESULTS_DIR, "cctv_bev_coordinates_mall.csv")
CSRNET_MODEL_PATH = os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth")

# 4) 공통 기타 설정
FPS = 15
DATASET_TYPE = "MALL"
MAX_FRAMES = 200 # 고속 검증용 프레임 수 제한 (2,000장 중 200프레임만 슬라이싱)

# =========================================================================
# 파이프라인 환경변수 구성 및 실행 함수
# =========================================================================
def run_step(step_name, script_name):
    print(f"\n" + "="*80)
    print(f"[STEP] {step_name} 실행 중...")
    print(f"File: {script_name}")
    print("="*80)
    
    script_abs_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_abs_path):
        print(f"[ERROR] 스크립트 파일을 찾을 수 없습니다: {script_abs_path}")
        sys.exit(1)

    # 개별 스크립트에 전달할 환경변수 매핑
    env = os.environ.copy()
    env["IMAGE_DIR"] = IMAGE_DIR
    env["OUTPUT_MP4"] = OUTPUT_MP4
    env["FPS"] = str(FPS)
    env["CSRNET_MODEL_PATH"] = CSRNET_MODEL_PATH
    env["CSR_RESULT_MP4"] = CSR_RESULT_MP4
    env["CCTV_BEV_CSV"] = CCTV_BEV_CSV
    env["LABEL_DIR"] = LABEL_DIR
    env["DATASET_TYPE"] = DATASET_TYPE
    env["MAX_FRAMES"] = str(MAX_FRAMES)

    # 윈도우 인코딩 오류 방지
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()
    
    # 프로세스 실행
    result = subprocess.run([PYTHON_EXE, script_abs_path], env=env)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[SUCCESS] {step_name} 완료! (소요 시간: {elapsed_time:.2f}초)")
    else:
        print(f"[FAILURE] {step_name} 실행 실패! (종료 코드: {result.returncode})")
        sys.exit(result.returncode)

# =========================================================================
# 메인 파이프라인 실행부
# =========================================================================
if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            # 콘솔의 코드 페이지를 UTF-8(65001)로 설정하여 문자 깨짐 방지
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    print("="*80)
    print(" [CCTV-Only Processing Pipeline Tester] ")
    print(f" Workspace: {BASE_DIR}")
    print(f" Python Exec: {PYTHON_EXE}")
    print("="*80)
    print(f" - Image Source: {IMAGE_DIR}")
    print(f" - Video Output: {OUTPUT_MP4}")
    print(f" - BEV CSV Dest: {CCTV_BEV_CSV}")
    print("="*80)

    # 1단계: 01_make_mp4.py (이미지 -> 비디오 생성)
    run_step("1단계: 비디오 변환 (01_make_mp4)", "01_make_mp4.py")

    # 2단계: 02_save_CSR_result.py (CSRNet 인원 밀도 추론 비디오 생성)
    run_step("2단계: CSRNet 밀도 추론 비디오 생성 (02_save_CSR_result)", "02_save_CSR_result.py")

    # 3단계: 03_video_to_bev_CSR.py (CSRNet 히트맵 -> BEV 2D 좌표 변환 CSV)
    run_step("3단계: CSRNet 기반 BEV 2D 좌표 추출 및 투영 (03_video_to_bev_CSR)", "03_video_to_bev_CSR.py")

    # 4단계: 04_cctv_spatial_analysis.py (CCTV 단독 공간 분석 및 백엔드 적재)
    run_step("4단계: CCTV 단독 공간 분석 및 백엔드 적재 (04_cctv_spatial_analysis)", "05_cctv_spatial_analysis.py")

    # 5단계: 09_aggregate_pedestrian_json.py (보행자 픽셀/3D BEV 좌표 JSON 집계)
    run_step("5단계: 보행자 픽셀 및 3D BEV 좌표 JSON 집계 (09_aggregate_pedestrian_json)", "09_aggregate_pedestrian_json.py")

    print("\n" + "="*80)
    print(" [CCTV-Only Pipeline Execution Complete] ")
    print(f" 최종 결과 CSV 좌표, JSON 집계 파일 및 클라우드 DB 적재가 정상적으로 완료되었습니다.")
    print("="*80)

