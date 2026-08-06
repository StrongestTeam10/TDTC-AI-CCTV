# db_connector.py
# Supabase PostgreSQL 직접 연결 (psycopg2) 및 신규 스키마(pedcord01h, mktsmry01s, crddnst01m, mrkrisk01m) 데이터 적재 모듈
import json
import os
import sys
import psycopg2
from psycopg2.extras import execute_values

def load_env():
    # cctv_ai_pipeline 루트 또는 상위 디렉터리의 .env 탐색 및 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "..", ".env"),
        os.path.join(current_dir, "..", "..", "..", ".env"),
        os.path.join(current_dir, "..", ".env")
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
            break

load_env()

DB_HOST = os.environ.get("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres.uusnthiedfwzlcsgtnhn")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "tdtcStrong10!")

def get_db_connection():
    """Supabase PostgreSQL 접속 커넥션 반환"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Supabase DB 커넥션 실패: {e}")
        return None

def bulk_insert_pedestrian_coordinates(data_list):
    """보행자 좌표 이력 (pedcord01h) 일괄 적재"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션] pedcord01h 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO pedcord01h 
            (zone_id, frame_id, person_id, pixel_x, pixel_y, bev_x_m, bev_y_m, risk_score, analysis_mode, video_id)
            VALUES %s
        """
        tuples = [
            (
                d.get('zone_id', 1),
                d.get('frame_id'),
                d.get('person_id'),
                d.get('pixel_x'),
                d.get('pixel_y'),
                d.get('bev_x_m'),
                d.get('bev_y_m'),
                d.get('risk_score', 0.0),
                d.get('analysis_mode', 'LIVE'),
                d.get('video_id', 1)
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] pedcord01h (보행자 좌표 이력) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] pedcord01h DB 적재 실패: {e}")
        if conn: conn.close()
        return False

def bulk_insert_external_factor(data_list):
    """CCTV 분석 외부요인 이력 (extfctr01h) 일괄 적재"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션 모드] extfctr01h 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO extfctr01h 
            (factor_id, market_id, video_id, target_date, weather_condition, temperature, event_category, updated_at)
            VALUES %s
            ON CONFLICT (factor_id) DO NOTHING;
        """
        tuples = [
            (
                d.get('factor_id', 1),
                d.get('market_id', 1),
                d.get('video_id', 1),
                d.get('target_date', '2026-07-31'),
                d.get('weather_condition', 'CLEAR'),
                d.get('temperature', 25.0),
                d.get('event_category', 'NORMAL'),
                d.get('updated_at', 'now()')
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] extfctr01h (외부 요인) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] extfctr01h DB 적재 실패: {e}")
        if conn: conn.close()
        return False

def bulk_insert_video_clip(data_list):
    """영상 마스터 테이블 (vdoclip01m) 일괄 적재"""
    if not data_list:
        return True
    
    # 0단계: FK 제약조건 만족을 위해 extfctr01h 외부요인(factor_id=1) 선제 등록 시도
    bulk_insert_external_factor([{'factor_id': 1, 'market_id': 1}])

    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션 모드] vdoclip01m 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO vdoclip01m 
            (clip_id, zone_id, factor_id, clip_type, s3_clip_url, start_time, end_time, expires_at, is_downloaded, is_deleted)
            VALUES %s
            ON CONFLICT (clip_id) DO NOTHING;
        """
        tuples = [
            (
                d.get('clip_id', 1),
                d.get('zone_id', 1),
                d.get('factor_id', 1),
                d.get('clip_type', 'LIVE'),
                d.get('s3_clip_url', 'https://s3.amazonaws.com/mangwon/cctv_clip_01.mp4'),
                d.get('start_time', 'now()'),
                d.get('end_time', 'now()'),
                d.get('expires_at', 'now()'),
                d.get('is_downloaded', False),
                d.get('is_deleted', False)
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] vdoclip01m (영상 마스터) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] vdoclip01m DB 적재 실패: {e}")
        if conn: conn.close()
        return False


def bulk_insert_pedestrian_coordinate_json(data_list, weather_info=None):
    """보행자 좌표 JSON 통합 테이블 (pedaggr01h), 외부요인(extfctr01h) 및 위험 점수 (mrkrisk01m) 연쇄 일괄 적재"""
    if not data_list:
        return True
    
    # 1단계: FK 제약조건 만족을 위해 extfctr01h 외부요인 적재
    if weather_info and 'extfctr_db_payload' in weather_info:
        ext_payload = weather_info['extfctr_db_payload']
        bulk_insert_external_factor([ext_payload])
    else:
        bulk_insert_external_factor([{'factor_id': 1, 'market_id': 1}])

    # 2단계: vdoclip01m 영상 클립 마스터 적재 (고유한 clip_id별로 추출하여 일괄 등록)
    unique_clips = {}
    for d in data_list:
        cid = d.get('clip_id')
        if cid and cid not in unique_clips:
            unique_clips[cid] = {
                'clip_id': cid,
                'zone_id': d.get('zone_id', 1),
                'factor_id': 1,
                'clip_type': 'LIVE',
                's3_clip_url': d.get('s3_clip_url', 'https://s3.amazonaws.com/mangwon/cctv_clip_01.mp4')
            }
    if unique_clips:
        bulk_insert_video_clip(list(unique_clips.values()))

    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션 모드] Supabase 미연결로 pedaggr01h / mrkrisk01m 적재 건수만 기록: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO pedaggr01h 
            (clip_id, zone_id, frame_id, video_id, pixels_json, bev_xyz_json, captured_at)
            VALUES %s
            RETURNING coord_id, frame_id;
        """
        tuples = [
            (
                d.get('clip_id', 1),
                d.get('zone_id', 1),
                d.get('frame_id'),
                d.get('video_id', 1),
                d.get('pixels_json', '{}'),
                d.get('bev_xyz_json', '{}'),
                d.get('captured_at', 'now()')
            ) for d in data_list
        ]
        
        # RETURNING coord_id 받아오기
        inserted_rows = execute_values(cursor, query, tuples, fetch=True)
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"[SUCCESS] pedaggr01h (보행자 좌표 JSON 통합) Supabase DB 적재 성공: {len(inserted_rows)} rows")
        
        # 3단계: 날씨 가중치(weather_weight) 적용하여 mrkrisk01m 위험 점수 적재 데이터 생성
        weather_weight = weather_info.get('weather_weight', 1.0) if weather_info else 1.0
        weather_label = weather_info.get('weather_label', 'CLEAR') if weather_info else 'CLEAR'
        print(f"[INFO] AI 위험도 산출 시 적용된 날씨 가중치: {weather_weight}x ({weather_label})")

        risk_list = []
        # inserted_rows와 data_list는 1대1로 순서가 완벽히 일치하므로 인덱스 매핑을 적용
        for i, (coord_id, frame_id) in enumerate(inserted_rows):
            if i >= len(data_list):
                break
            ref_d = data_list[i]
            
            # 레코드에서 total_count 직접 추출
            if 'total_count' in ref_d:
                count = ref_d['total_count']
            else:
                pixels_str = ref_d.get('pixels_json', '{}')
                try:
                    p_map = json.loads(pixels_str) if isinstance(pixels_str, str) else pixels_str
                    count = len(p_map) if p_map else 0
                except Exception:
                    count = 0
            
            # 3. 입구 인파 유입량 급증 (Inflow Spike) 감지 로직
            # 평시 평균 입구 유입자 수 = 15명 기준, 2배인 30명 이상 시 급증 경보 발령
            base_score = count * 3.5
            final_score = round(min(100.0, base_score * weather_weight), 1)
            
            inflow_spike = (count >= 30) or (count >= 15 and weather_info and weather_info.get('inflow_monitor_active'))
            
            if inflow_spike and weather_info and weather_info.get('inflow_monitor_active'):
                level = "HIGH"
                reason = "RAIN_PREDICTION_INFLOW_SPIKE"
            else:
                level = "SAFE" if final_score < 40 else "WARN" if final_score < 70 else "HIGH"
                reason = weather_info.get('reason_code', 'NORMAL') if weather_info else "NORMAL"
            
            risk_list.append({
                'coord_id': coord_id,
                'risk_score': final_score,
                'risk_level': level,
                'reason_code': reason,
                'total_count': count,
                'detected_at': ref_d.get('captured_at', 'now()')
            })

            
        bulk_insert_risk_score(risk_list)

        return True
    except Exception as e:
        print(f"[ERROR] pedaggr01h DB 적재 실패: {e}")
        if conn: conn.close()
        return False




def bulk_insert_market_summary(data_list):
    """시장 전체 통계 요약 (mktsmry01s) 일괄 적재"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션] mktsmry01s 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO mktsmry01s 
            (market_id, frame_id, total_cctv_count, avg_density_score, max_density_score, max_risk_score, analysis_mode, video_id)
            VALUES %s
        """
        tuples = [
            (
                d.get('market_id', 1),
                d.get('frame_id'),
                d.get('total_cctv_count'),
                d.get('avg_density_score'),
                d.get('max_density_score'),
                d.get('max_risk_score'),
                d.get('analysis_mode', 'LIVE'),
                d.get('video_id', 1)
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] mktsmry01s (시장 통계 요약) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] mktsmry01s DB 적재 실패: {e}")
        if conn: conn.close()
        return False

def bulk_insert_crowd_density(data_list):
    """인구 밀집도 현재 (crddnst01m) 및 로그 (crddnst01h) 일괄 적재"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션] crddnst01m / crddnst01h 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        
        # 1. crddnst01m (현재 밀집도)
        query_m = """
            INSERT INTO crddnst01m 
            (zone_id, visitor_count, density_score, status_level, analysis_mode, video_id, frame_id, captured_at)
            VALUES %s
        """
        tuples_m = [
            (
                d.get('zone_id', 1),
                d.get('visitor_count'),
                d.get('density_score'),
                d.get('status_level', 'NORMAL'),
                d.get('analysis_mode', 'LIVE'),
                d.get('video_id', 1),
                d.get('frame_id'),
                'now()'
            ) for d in data_list
        ]
        execute_values(cursor, query_m, tuples_m)

        # 2. crddnst01h (밀집도 로그)
        query_h = """
            INSERT INTO crddnst01h 
            (crowd_density_id, visitor_count, density_score, status_level, analysis_mode, video_id, frame_id, captured_at)
            VALUES %s
        """
        tuples_h = [
            (
                d.get('zone_id', 1), # crowd_density_id대신 zone_id 참조 매핑
                d.get('visitor_count'),
                d.get('density_score'),
                d.get('status_level', 'NORMAL'),
                d.get('analysis_mode', 'LIVE'),
                d.get('video_id', 1),
                d.get('frame_id'),
                'now()'
            ) for d in data_list
        ]
        execute_values(cursor, query_h, tuples_h)

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] crddnst01m / crddnst01h (밀집도 및 로그) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] crddnst01m / crddnst01h DB 적재 실패: {e}")
        if conn: conn.close()
        return False

def bulk_insert_risk_score(data_list):
    """위험 점수 (mrkrisk01m) 일괄 적재 (최신 스키마: coord_id FK 연동)"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션 모드] mrkrisk01m 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO mrkrisk01m 
            (coord_id, risk_score, risk_level, reason_code, total_count, detected_at)
            VALUES %s
        """
        tuples = [
            (
                d.get('coord_id'),
                d.get('risk_score', 0.0),
                d.get('risk_level', 'SAFE'),
                d.get('reason_code', 'NORMAL'),
                d.get('total_count', 0),
                d.get('detected_at', 'now()')
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] mrkrisk01m (위험 점수 이력) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] mrkrisk01m DB 적재 실패: {e}")
        if conn: conn.close()
        return False


def bulk_insert_emergency_alert(data_list):
    """긴급 신고 이력 (emgalrt01h) 일괄 적재"""
    if not data_list:
        return True
    conn = get_db_connection()
    if not conn:
        print(f"[INFO] [DB 시뮬레이션] emgalrt01h 적재 건수: {len(data_list)} rows")
        return True
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO emgalrt01h 
            (zone_id, risk_id, alert_type, s3_clip_url, is_resolved, alerted_at)
            VALUES %s
        """
        tuples = [
            (
                d.get('zone_id', 1),
                d.get('risk_id', 1),
                d.get('alert_type', 'HIGH_RISK_SURGE'),
                d.get('s3_clip_url', ''),
                d.get('is_resolved', False),
                'now()'
            ) for d in data_list
        ]
        execute_values(cursor, query, tuples)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[SUCCESS] emgalrt01h (긴급 알람 이력) Supabase DB 적재 성공: {len(data_list)} rows")
        return True
    except Exception as e:
        print(f"[ERROR] emgalrt01h DB 적재 실패: {e}")
        if conn: conn.close()
        return False

# 이전 함수들과의 하위 호환성 유지 래퍼 (Wrapper)
def bulk_insert_summary(data_list):
    return bulk_insert_market_summary(data_list)

def bulk_insert_history(data_list):
    return bulk_insert_pedestrian_coordinates(data_list)

def bulk_insert_roi_log(data_list):
    return bulk_insert_crowd_density(data_list)
