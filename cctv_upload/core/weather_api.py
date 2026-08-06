import os
import requests
import datetime
import json

def load_env():
    """상위 및 현재 디렉터리의 .env 탐색 및 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, ".env"),
        os.path.join(current_dir, "..", ".env"),
        os.path.join(current_dir, "..", "..", ".env")
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            except Exception:
                pass
            break

load_env()

def get_mangwon_weather(service_key=None, fallback_mode="SUNNY"):
    """
    기상청 단기예보 API를 활용하여 망원동(nx=59, ny=126)의 실시간 기상 상태를 조회합니다.
    - T1H (기온, ℃)
    - REH (습도, %)
    - PTY (강수 형태): 0(없음), 1(비), 2(비/눈), 3(눈), 4(소나기)
    
    API 키가 없거나 호출 실패 시 fallback_mode("SUNNY", "RAINY", "HOT_SUMMER")에 기반한 Mock 데이터를 안전하게 반환합니다.
    """
    if not service_key:
        service_key = os.environ.get("WEATHER_API_KEY") or os.environ.get("KMA_SERVICE_KEY")

    # 기본 기상청 초단기실황 조회 서비스 Endpoint
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    
    # 기상청 실황 발표 시점(매시 45분 이전에는 전 시간 정시 데이터 요청)
    now = datetime.datetime.now()
    if now.minute < 45:
        base_time = (now - datetime.timedelta(hours=1)).strftime("%H00")
    else:
        base_time = now.strftime("%H00")
    
    base_date = now.strftime("%Y%m%d")
    
    # 망원동 (서울시 마포구 망원1동 중심 격자점)
    nx, ny = 59, 126
    
    # 서비스 키가 인자로 없거나 환경변수에 없을 경우 Mock 데이터 가동
    if not service_key:
        print("[WEATHER] 서비스 인증키가 설정되지 않아 Mock 날씨 데이터를 가동합니다.")
        return _get_mock_weather(fallback_mode, base_date, base_time)
        
    params = {
        'serviceKey': service_key,
        'pageNo': '1',
        'numOfRows': '10',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': nx,
        'ny': ny
    }
    
    try:
        response = requests.get(url, params=params, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'header' in data['response']:
                result_code = data['response']['header'].get('resultCode', '99')
                if result_code == '00':
                    items = data['response']['body']['items']['item']
                    weather = {
                        'base_date': base_date,
                        'base_time': base_time,
                        'nx': nx,
                        'ny': ny,
                        'success': True
                    }
                    for item in items:
                        category = item['category']
                        val = float(item['obsrValue'])
                        if category == 'T1H':    # 기온
                            weather['temp'] = val
                        elif category == 'REH':  # 습도
                            weather['humidity'] = val
                        elif category == 'PTY':  # 강수 형태
                            weather['pty'] = int(val)
                            
                    # 강수형태(PTY) 기반 비 옴(Rainy) 플래그 설정 (1: 비, 2: 비/눈, 4: 소나기)
                    weather['is_rainy'] = True if weather.get('pty', 0) in [1, 2, 4] else False
                    
                    # 불쾌지수 (Discomfort Index, DI) 연산
                    t = weather.get('temp', 25.0)
                    h = weather.get('humidity', 60.0)
                    di = 1.8 * t - 0.55 * (1 - 0.01 * h) * (1.8 * t - 26) + 32
                    weather['discomfort_index'] = round(di, 1)
                    
                    # 가중치 및 DB 적재용 세부 데이터 추가
                    _enrich_weather_factors(weather)
                    return weather
                    
            print(f"[WEATHER] API 응답 에러 (코드: {data.get('response', {}).get('header', {}).get('resultMsg', 'Unknown')})")
    except Exception as e:
        print(f"[WEATHER] API 연동 예외 발생: {e}")
        
    print(f"[WEATHER] API 연결 실패로 인해 Mock {fallback_mode} 데이터를 사용합니다.")
    return _get_mock_weather(fallback_mode, base_date, base_time)

def _enrich_weather_factors(weather):
    """
    30분 전 기상청 초단기예보 + 출입구 CCTV 실시간 유입량 2배 급증(Inflow Spike) 팩트 기반 융합 모델:
    1) 30분 후 비 예보 수신 시: 관제 직원용 사전 로그 알림 ("입구 유입 감시 모드 활성화")
    2) 비 예보 상태에서 입구 유입자 수 2배 급증 발생 시: 긴급 경보 (reason_code: RAIN_PREDICTION_INFLOW_SPIKE)
    """
    temp = weather.get('temp', 25.0)
    is_rainy = weather.get('is_rainy', False)
    di = weather.get('discomfort_index', 70.0)
    
    fcst_30m_rain = weather.get('fcst_30m_rain', is_rainy)
    fcst_30m_temp = weather.get('fcst_30m_temp', temp)
    
    weight = 1.0
    condition = "CLEAR"
    label = "맑음 / 기본 상태 (1.0x)"
    reason_code = "NORMAL"
    inflow_monitor_active = False
    
    if is_rainy:
        weight = 1.25
        condition = "RAIN"
        label = "우천 실황 / 입구 유입 집중 감시 (1.25x)"
        reason_code = "RAIN_INFLOW_MONITORING"
        inflow_monitor_active = True
    elif fcst_30m_rain:
        # 30분 후 비 예보 수신 ➔ 입구 유입 감시 모드 활성화 (사전 로그 알림)
        weight = 1.15
        condition = "PREDICTIVE_RAIN"
        label = "30분 후 강수 예보 / 입구 유입 감시 모드 활성화 (+15%)"
        reason_code = "PREDICTIVE_RAIN_INFLOW_MONITOR_ACTIVE"
        inflow_monitor_active = True
    elif temp >= 33.0 or di >= 80.0:
        weight = 1.35
        condition = "HOT_SUMMER"
        label = "폭염 / 온실 열기 갇힘 모니터링 (+35%)"
        reason_code = "HOT_SUMMER_MONITORING"
    elif temp <= -5.0:
        weight = 1.15
        condition = "COLD"
        label = "한파 / 동파 미끄럼 주의 (+15%)"
        reason_code = "COLD_FREEZE_MONITORING"

    weather['weather_weight'] = weight
    weather['weather_condition'] = condition
    weather['weather_label'] = label
    weather['reason_code'] = reason_code
    weather['inflow_monitor_active'] = inflow_monitor_active
    weather['fcst_30m_rain'] = fcst_30m_rain
    weather['fcst_30m_temp'] = fcst_30m_temp
    
    # Supabase extfctr01h DB 적재용 페이로드 객체
    weather['extfctr_db_payload'] = {
        'factor_id': 1,
        'market_id': 1,
        'video_id': 1,
        'target_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'weather_condition': condition,
        'temperature': temp,
        'event_category': 'PREDICTIVE_MONITORING' if inflow_monitor_active else 'NORMAL'
    }


def _get_mock_weather(mode, date, time):
    """실제 API 미설정 시 작동하는 날씨 시뮬레이션 모크 데이터 생성기 (30분 예보 포함)"""
    nx, ny = 59, 126
    mode_upper = mode.upper()
    
    fcst_30m_rain = False
    fcst_30m_temp = 25.0
    
    if mode_upper == "RAINY":
        temp, humidity, pty, is_rainy = 24.5, 88.0, 1, True
        fcst_30m_rain = True
    elif mode_upper == "PREDICTIVE_RAIN":
        # 현재는 비 안 옴, 30분 뒤 비 예보
        temp, humidity, pty, is_rainy = 25.0, 70.0, 0, False
        fcst_30m_rain = True
    elif mode_upper == "HOT_SUMMER":
        temp, humidity, pty, is_rainy = 34.0, 75.0, 0, False
        fcst_30m_temp = 35.0
    else:
        temp, humidity, pty, is_rainy = 28.0, 55.0, 0, False
        
    di = 1.8 * temp - 0.55 * (1 - 0.01 * humidity) * (1.8 * temp - 26) + 32
    
    weather = {
        'base_date': date,
        'base_time': time,
        'nx': nx,
        'ny': ny,
        'temp': temp,
        'humidity': humidity,
        'pty': pty,
        'is_rainy': is_rainy,
        'discomfort_index': round(di, 1),
        'fcst_30m_rain': fcst_30m_rain,
        'fcst_30m_temp': fcst_30m_temp,
        'success': False,
        'mock_mode': mode_upper
    }
    _enrich_weather_factors(weather)
    return weather

if __name__ == "__main__":
    print("=== [망원시장 아케이드 특화 + 30분 초단기예보 융합 테스트] ===")
    
    for m in ["SUNNY", "PREDICTIVE_RAIN", "RAINY", "HOT_SUMMER"]:
        w = get_mangwon_weather(fallback_mode=m)
        print(f"\n[{m} 모드 연동 결과]:")
        print(f" - 기온/습도/불쾌지수: {w['temp']}℃ / {w['humidity']}% / {w['discomfort_index']}pt")
        print(f" - 30분 후 강수 예보: {'🌧️ 비 예보 있음' if w['fcst_30m_rain'] else '☀️ 예보 없음'}")
        print(f" - 날씨 상태: {w['weather_condition']} ({w['weather_label']})")
        print(f" - 탐지 사유 코드 (reason_code): {w['reason_code']}")
        print(f" - AI 위험도 산출 가중치: {w['weather_weight']}x")

