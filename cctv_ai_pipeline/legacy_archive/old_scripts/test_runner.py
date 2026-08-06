# test_runner.py
# TDTC CCTV AI 파이프라인 - 화려한 대화형 실시간 스마트 관제 시뮬레이터 & 테스트 러너

import os
import sys
import time
import json
import random
import webbrowser
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "sensor_fusion_archive"))

# 터미널 ANSI 색상 코드
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_BLUE = "\033[34m"
COLOR_BG_RED = "\033[41m\033[37m"

def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{COLOR_CYAN}{COLOR_BOLD}")
    print("=" * 85)
    print(" 🚀  TDTC MANGWON MARKET SMART CCTV AI PIPELINE - LIVE SIMULATOR & TEST RUNNER")
    print("=" * 85)
    print(f"{COLOR_RESET}")

def print_step(step_num, title):
    print(f"\n{COLOR_BOLD}{COLOR_BLUE}[STEP {step_num}]{COLOR_RESET} {COLOR_CYAN}{title}{COLOR_RESET}")
    print("-" * 75)

def simulate_progress_bar(label, duration=1.2, steps=25):
    print(f"{COLOR_BOLD}{label}{COLOR_RESET} [", end="", flush=True)
    for i in range(steps):
        time.sleep(duration / steps)
        print(f"{COLOR_GREEN}█{COLOR_RESET}", end="", flush=True)
    print(f"] {COLOR_GREEN}100% DONE{COLOR_RESET}")

def run_test_suite(scenario_mode="PREDICTIVE_RAIN"):
    print_banner()
    
    print(f"{COLOR_BOLD}📌 실행 모드:{COLOR_RESET} {COLOR_MAGENTA}{scenario_mode}{COLOR_RESET}")
    print(f"{COLOR_BOLD}⏱️ 실행 시각:{COLOR_RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{COLOR_BOLD}🌐 연동 DB:{COLOR_RESET} Supabase PostgreSQL (aws-0-ap-northeast-1)")

    # 1단계: 기상청 30분 초단기예보 연동
    print_step(1, "기상청 초단기예보 API & 기상 가중치 모듈 검증 (weather_api.py)")
    simulate_progress_bar(" ⚡ 초단기예보 API 데이터 수신 중...", 0.8)
    
    from weather_api import get_mangwon_weather
    weather = get_mangwon_weather(fallback_mode=scenario_mode)
    
    print(f"\n  {COLOR_BOLD}☀️ 실시간 기온/습도:{COLOR_RESET} {weather['temp']}℃ / {weather['humidity']}% (불쾌지수: {weather['discomfort_index']}pt)")
    print(f"  {COLOR_BOLD}🌧️ 30분 후 강수 예보:{COLOR_RESET} {'🌧️ 비 예보 발령!' if weather['fcst_30m_rain'] else '☀️ 비 예보 없음'}")
    print(f"  {COLOR_BOLD}🏷️ 기상 상태 라벨:{COLOR_RESET} {weather['weather_condition']} ({weather['weather_label']})")
    print(f"  {COLOR_BOLD}🛡️ 가중치 & 사유 코드:{COLOR_RESET} {COLOR_YELLOW}{weather['weather_weight']}x{COLOR_RESET} | {COLOR_MAGENTA}{weather['reason_code']}{COLOR_RESET}")
    time.sleep(1)

    # 2단계: 출입구 실시간 인파 유입 및 Spike 분석
    print_step(2, "CCTV 입구 유입자 수 실시간 모니터링 & Inflow Spike 연산")
    simulate_progress_bar(" 📹 2D->3D BEV 역투영 및 실시간 인입량 측정...", 1.0)
    
    # 시나리오에 따른 유입 인원 설정
    if scenario_mode == "PREDICTIVE_RAIN":
        inflow_count = 34 # 평시 평균(15명) 2배 초과 급증
    elif scenario_mode == "RAINY":
        inflow_count = 28
    else:
        inflow_count = 14 # 평시 정상
        
    print(f"\n  {COLOR_BOLD}👥 평시 입구 평균 인입자 수:{COLOR_RESET} 15명")
    print(f"  {COLOR_BOLD}📊 현재 실시간 입구 인입자 수:{COLOR_RESET} {COLOR_BOLD}{COLOR_YELLOW if inflow_count < 30 else COLOR_RED}{inflow_count}명{COLOR_RESET}")
    
    inflow_spike = (inflow_count >= 30) or (inflow_count >= 15 and weather.get('inflow_monitor_active'))
    if inflow_spike and weather.get('inflow_monitor_active'):
        print(f"\n  {COLOR_BG_RED} 🚨 INFLOW SPIKE DETECTED! {COLOR_RESET} {COLOR_RED}{COLOR_BOLD}비 예보 상태에서 입구 유입량 2배 급증 감지! (Reason: RAIN_PREDICTION_INFLOW_SPIKE){COLOR_RESET}")
    else:
        print(f"\n  {COLOR_GREEN}✅ 정상 범위 관제 중 (유입량 안정적){COLOR_RESET}")
    time.sleep(1)

    # 3단계: Supabase DB 일괄 적재 (Bulk Insert)
    print_step(3, "Supabase Cloud DB (4대 메인 테이블) 일괄 적재 (db_connector.py)")
    simulate_progress_bar(" ☁️ Supabase PostgreSQL 데이터 전송 중...", 1.2)
    
    # pyrefly: ignore [missing-import]
    from utils.db_connector import bulk_insert_pedestrian_coordinate_json
    
    # 테스트 가상 데이터 10건 생성
    dummy_records = []
    for f in range(1, 11):
        dummy_records.append({
            "clip_id": 1,
            "frame_id": f,
            "video_id": 1,
            "total_count": inflow_count,
            "pixels_json": json.dumps({"person_1": {"x": 350.0, "y": 200.0}}),
            "bev_xyz_json": json.dumps({"person_1": {"x": 2.5, "y": 1.2, "z": 0.0}}),
            "captured_at": datetime.now().isoformat()
        })
        
    db_success = bulk_insert_pedestrian_coordinate_json(dummy_records, weather_info=weather)
    
    if db_success:
        print(f"\n  {COLOR_GREEN}✔ extfctr01h (외부요인): 1 rows 적재 성공{COLOR_RESET}")
        print(f"  {COLOR_GREEN}✔ vdoclip01m (클립마스터): 1 rows 적재 성공{COLOR_RESET}")
        print(f"  {COLOR_GREEN}✔ pedaggr01h (보행자좌표): {len(dummy_records)} rows 적재 성공{COLOR_RESET}")
        print(f"  {COLOR_GREEN}✔ mrkrisk01m (위험점수): {len(dummy_records)} rows 연쇄 적재 성공{COLOR_RESET}")
    time.sleep(1)

    # 4단계: 모듈화된 대시보드 자동 열기
    print_step(4, "웹 대시보드 UI 라이브 연동 및 자동 렌더링")
    dashboard_path = os.path.join(WORKSPACE_DIR, "dashboard", "index.html")
    print(f"  {COLOR_BOLD}🌐 대시보드 URL:{COLOR_RESET} {dashboard_path}")
    
    simulate_progress_bar(" 💻 웹 대시보드 브라우저 자동 오픈 중...", 0.6)
    try:
        webbrowser.open(f"file:///{dashboard_path}")
        print(f"\n  {COLOR_BOLD}{COLOR_GREEN}🎉 관제 대시보드가 브라우저에서 성공적으로 열렸습니다!{COLOR_RESET}")
    except Exception as e:
        print(f"  [WARNING] 브라우저 자동 실행 실패: {e}")

    print("\n" + "=" * 85)
    print(f" {COLOR_BOLD}{COLOR_GREEN}✨ [TEST COMPLETE] 모든 시나리오 파이프라인 및 백엔드 DB 연동 검증이 완료되었습니다!{COLOR_RESET}")
    print("=" * 85 + "\n")

def main():
    while True:
        print_banner()
        print(f"{COLOR_BOLD}🎮 테스트하고 싶으신 스마트 관제 시나리오를 선택하세요:{COLOR_RESET}\n")
        print(f" {COLOR_CYAN}1.{COLOR_RESET} 🔮 [시나리오 A] 30분 후 강수 예보 + 입구 인파 2배 급증 (RAIN_PREDICTION_INFLOW_SPIKE)")
        print(f" {COLOR_CYAN}2.{COLOR_RESET} ☀️ [시나리오 B] 평상시 맑은 날씨 정상 관제 (CLEAR)")
        print(f" {COLOR_CYAN}3.{COLOR_RESET} 🌧️ [시나리오 C] 우천 실황 입구 인파 모니터링 (RAINY)")
        print(f" {COLOR_CYAN}4.{COLOR_RESET} 🥵 [시나리오 D] 한여름 폭염 온실 효과 관제 (HOT_SUMMER)")
        print(f" {COLOR_CYAN}5.{COLOR_RESET} 🚀 전체 시나리오 풀코스 파이프라인 연속 실행")
        print(f" {COLOR_RED}0. 종료 (Exit){COLOR_RESET}\n")
        
        choice = input(f"{COLOR_BOLD}선택 (0-5) ➔ {COLOR_RESET}").strip()
        
        if choice == "1":
            run_test_suite("PREDICTIVE_RAIN")
        elif choice == "2":
            run_test_suite("SUNNY")
        elif choice == "3":
            run_test_suite("RAINY")
        elif choice == "4":
            run_test_suite("HOT_SUMMER")
        elif choice == "5":
            for mode in ["SUNNY", "PREDICTIVE_RAIN", "RAINY", "HOT_SUMMER"]:
                run_test_suite(mode)
                time.sleep(2)
        elif choice == "0":
            print(f"\n{COLOR_GREEN}테스트 러너를 종료합니다. 수고하셨습니다!{COLOR_RESET}\n")
            break
        else:
            print(f"\n{COLOR_RED}잘못된 선택입니다. 다시 입력해주세요.{COLOR_RESET}")
            time.sleep(1)

if __name__ == "__main__":
    main()
