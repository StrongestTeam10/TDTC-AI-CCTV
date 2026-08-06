# run_pipeline.py: CCTV 프레임 처리, CSRNet/YOLO 기반 인원 감지, 센서 퓨전 및 검증 등 전체 파이프라인 단계를 통합적으로 실행하고 관리하는 메인 스크립트입니다.
import os
import sys
import subprocess
import time

# =========================================================================
# [설정] 통합 경로 및 파라미터 정의 (경로 변경 시 여기만 수정하세요)
# =========================================================================
BASE_DIR = r"E:\AIVLE_10team"

# 실행할 파이썬 가상환경 인터프리터 경로
# 패키지(cv2, ultralytics, torch 등)가 완벽히 설치된 가상환경 파이썬 경로를 지정합니다.
PYTHON_EXE = os.environ.get("PIPELINE_PYTHON", r"E:\anaconda\envs\lidar_env\python.exe")
if not os.path.exists(PYTHON_EXE):
    # 폴백: 환경에 지정되어 있지 않으면 현재 인터프리터 사용
    PYTHON_EXE = sys.executable

# 1) 입력 데이터 경로
IMAGE_DIR = r"E:\test\cctv_cafe"                # CCTV 프레임 이미지 폴더
LABEL_DIR = r"E:\test\cctv_cafe_label"          # GT 정답 JSON 라벨 폴더

# 2) 출력 및 결과물 저장 경로
OUTPUT_DIR = r"E:\test"                          # 변환된 비디오(.mp4)가 저장될 폴더
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "cctv_cafe_output.mp4")
CSR_RESULT_MP4 = os.path.join(OUTPUT_DIR, "cctv_cafe_result.mp4")

# 3) 결과 CSV 및 모델 가중치 경로
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CCTV_BEV_CSV = os.path.join(RESULTS_DIR, "cctv_bev_coordinates.csv")
FUSION_SUMMARY_CSV = os.path.join(RESULTS_DIR, "sensor_fusion_summary_v2.csv")
CSRNET_MODEL_PATH = os.path.join(RESULTS_DIR, "csrnet_ultimate_epoch_8.pth")

# 4) 공통 기타 설정
FPS = 15

# =========================================================================
# 파이프라인 환경변수 구성 및 실행 함수
# =========================================================================
def run_step(step_name, script_rel_path):
    print(f"\n" + "="*80)
    print(f"[STEP] {step_name} 실행 중...")
    print(f"File: {script_rel_path}")
    print("="*80)
    
    script_abs_path = os.path.join(BASE_DIR, script_rel_path)
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
    env["FUSION_SUMMARY_CSV"] = FUSION_SUMMARY_CSV
    env["LABEL_DIR"] = LABEL_DIR

    # 윈도우 인코딩 오류 방지
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()
    
    # 프로세스 실행 (지정된 가상환경 파이썬 사용)
    result = subprocess.run([PYTHON_EXE, script_abs_path], env=env)
    
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[SUCCESS] {step_name} 완료! (소요 시간: {elapsed_time:.2f}초)")
    else:
        print(f"[FAILURE] {step_name} 실행 실패! (종료 코드: {result.returncode})")
        sys.exit(result.returncode)

# =========================================================================
# 메인 파이프라인 루프
# =========================================================================
if __name__ == "__main__":
    # 한국 윈도우 터미널 출력(UTF-8) 강제 지시 설정 가능
    if sys.platform == "win32":
        try:
            import msvcrt
            import ctypes
            # 콘솔의 코드 페이지를 UTF-8(65001)로 설정하여 문자 깨짐 방지
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    print("="*80)
    print(" [CCTV + LiDAR Sensor Fusion Pipeline Integrated Tester] ")
    print(f" Workspace: {BASE_DIR}")
    print(f" Python Exec: {PYTHON_EXE}")
    print("="*80)
    print(f" - Image Source: {IMAGE_DIR}")
    print(f" - Label Source: {LABEL_DIR}")
    print(f" - Video Output: {OUTPUT_MP4}")
    print(f" - Final CSV:    {FUSION_SUMMARY_CSV}")
    print("="*80)

    # 1단계: make_mp4.py (이미지 -> 비디오 생성)
    run_step("1단계: 비디오 변환 (make_mp4)", os.path.join("4_inference_vis", "make_mp4.py"))

    # 2단계: save_CSR_result.py (CSRNet 인원 밀도 추론 비디오 생성)
    run_step("2단계: CSRNet 밀도 추론 비디오 생성 (save_CSR_result)", os.path.join("4_inference_vis", "save_CSR_result.py"))

    # 3단계: video_to_bev_CSR.py (CSRNet 히트맵 -> BEV 2D 좌표 변환 CSV)
    run_step("3단계: CSRNet 기반 BEV 2D 좌표 추출 (video_to_bev_CSR)", os.path.join("3_preprocessing", "video_to_bev_CSR.py"))

    # 4단계: fusion_validation_v3.py (LiDAR 히트맵 추론 및 CCTV BEV 격자 가중치 퓨전)
    run_step("4단계: 센서 퓨전 검증 v3 (fusion_validation_v3)", os.path.join("5_validation_test", "fusion_validation_v3.py"))

    # 5단계: evaluate_with_labels.py (GT 정답 데이터를 기준으로 한 정량적 정밀 평가 실행)
    run_step("5단계: GT 라벨 대비 종합 평가 (evaluate_with_labels)", os.path.join("6_evaluation", "evaluate_with_labels.py"))

    print("\n" + "="*80)
    print(" [Pipeline Integrated Test Complete] ")
    print(f" Results saved in:")
    print(f"   - Evaluation Summary: {os.path.join(RESULTS_DIR, 'label_evaluation_summary.csv')}")
    print("="*80)
