-- schema_ddl.sql
-- CCTV & LiDAR 센서 퓨전 적재용 Supabase 데이터베이스 테이블 스키마 DDL

-- 1. 프레임별 센서 퓨전 요약 (sf_frame_summary)
CREATE TABLE IF NOT EXISTS sf_frame_summary (
    frame_id INT PRIMARY KEY,
    timestamp_sec DOUBLE PRECISION NOT NULL,
    cctv_count INT NOT NULL,
    lidar_added_count INT NOT NULL,
    total_fusion_count INT NOT NULL,
    max_risk_score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. 보행자별 실시간 BEV 위치 이력 (sf_pedestrian_history)
CREATE TABLE IF NOT EXISTS sf_pedestrian_history (
    id BIGSERIAL PRIMARY KEY,
    frame_id INT NOT NULL,
    person_id INT NOT NULL,
    bev_x_m DOUBLE PRECISION NOT NULL,
    bev_y_m DOUBLE PRECISION NOT NULL,
    detection_source VARCHAR(16) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. 구역별 위험도 통계 (sf_roi_risk_log)
CREATE TABLE IF NOT EXISTS sf_roi_risk_log (
    log_id BIGSERIAL PRIMARY KEY,
    frame_id INT NOT NULL,
    roi_id INT NOT NULL,
    roi_name VARCHAR(64) NOT NULL,
    people_in_roi INT NOT NULL,
    occupancy_intensity DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. 인덱스 설정 (빠른 실시간 관제 및 대시보드 조회를 위한 인덱스 최적화)
CREATE INDEX IF NOT EXISTS idx_sf_pedestrian_frame ON sf_pedestrian_history(frame_id);
CREATE INDEX IF NOT EXISTS idx_sf_roi_risk_frame ON sf_roi_risk_log(frame_id);
CREATE INDEX IF NOT EXISTS idx_sf_frame_summary_time ON sf_frame_summary(created_at DESC);
