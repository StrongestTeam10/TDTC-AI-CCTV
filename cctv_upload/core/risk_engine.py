# ===================================================================================================
# [risk_validation_engine.py - AI 위험 감지 & 오탐 방지 검증 엔진]
# 1. YOLO 객체 탐지 Confidence Score(>=0.5) 필터링으로 1차 노이즈를 제거합니다.
# 2. 1~2프레임 순간 튐 수치를 차단하고, 3초(30프레임) 이상 연속 위험 유지 시에만 최종 출동 알람을 발령합니다.
# ===================================================================================================

import time
import sys
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class RiskValidator:
    """
    YOLO 객체 탐지 Confidence 필터링 및 연속 감지 프레임 기반 오탐 방지 검증 엔진
    """
    def __init__(self, 
                 conf_threshold: float = 0.5,
                 evac_score_threshold: float = 70.0,
                 alarm_delay_sec: float = 3.0,
                 fps: float = 10.0):
        """
        :param conf_threshold: YOLO 객체 탐지 최소 신뢰도 (0.5 이상만 사용)
        :param evac_score_threshold: 대피/출동 경보 위험도 점수 임계값 (기본 70pt)
        :param alarm_delay_sec: 출동 알람 발령에 필요한 최소 지속 시간 (기본 3.0초)
        :param fps: 동영상/스트림 프레임 레이트 (기본 10.0 FPS)
        """
        self.conf_threshold = conf_threshold
        self.evac_score_threshold = evac_score_threshold
        self.alarm_delay_sec = alarm_delay_sec
        self.fps = fps
        
        # 지속시간에 필요한 최소 연속 프레임 수 연산 (3.0초 * 10 FPS = 30 프레임)
        self.required_alarm_frames = max(1, int(round(alarm_delay_sec * fps)))
        
        # 내부 프레임 카운터 상태 관리
        self.evacuation_frames_counter = 0
        self.is_dispatch_active = False

    def filter_detections(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. Confidence Score 필터링: 신뢰도 미달 객체 제거 (오탐 1차 방지)
        """
        valid_detections = []
        for det in raw_detections:
            conf = det.get('confidence', 1.0)
            if conf >= self.conf_threshold:
                valid_detections.append(det)
        return valid_detections

    def update_and_validate(self, 
                            current_cri_score: float, 
                            raw_detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        2. 프레임 단위 위험도 및 연속 감지 프레임 검증 연산
        :param current_cri_score: 현재 프레임의 종합 위험 지수 (CRI)
        :param raw_detections: 감지된 객체 리스트
        :return: 검증된 상태 및 경보 정보 딕셔너리
        """
        # (1) Detection Confidence 필터링 수행
        valid_dets = self.filter_detections(raw_detections) if raw_detections else []

        # (2) 연속 감지 프레임 디바운싱 연산 (노이즈 방지)
        if current_cri_score >= self.evac_score_threshold:
            # 위험 수치 이상일 때 프레임 카운터 증가 (최대 필요 프레임 한도로 제한)
            self.evacuation_frames_counter = min(
                self.required_alarm_frames, 
                self.evacuation_frames_counter + 1
            )
        else:
            # 수치가 내려가면 리셋 대신 점진적 감쇄 (1씩 감소)하여 노이즈로 인한 경보 깜빡임 방지
            self.evacuation_frames_counter = max(0, self.evacuation_frames_counter - 1)

        # (3) 최종 출동 경보 발령 여부 판정
        self.is_dispatch_active = (self.evacuation_frames_counter >= self.required_alarm_frames)

        # (4) 관제 상태 등급 결정
        if current_cri_score >= self.evac_score_threshold:
            if self.is_dispatch_active:
                status_code = "DISPATCH_ACTIVE"
                status_label = "[DISPATCH_ACTIVE] EMERGENCY DISPATCH ACTIVE (관제소 출동 신호 발생)"
            else:
                elapsed_sec = round(self.evacuation_frames_counter / self.fps, 1)
                status_code = "EVACUATING_PENDING"
                status_label = f"[WARNING] 위험 지속 감지 중 ({elapsed_sec}s / {self.alarm_delay_sec}s)"
        elif current_cri_score >= (self.evac_score_threshold * 3.0 / 7.0):
            status_code = "CONGESTED"
            status_label = "[CONGESTED] 혼잡"
        else:
            status_code = "NORMAL"
            status_label = "[NORMAL] 정상"

        return {
            "status_code": status_code,
            "status_label": status_label,
            "is_dispatch_active": self.is_dispatch_active,
            "evacuation_frames_counter": self.evacuation_frames_counter,
            "required_alarm_frames": self.required_alarm_frames,
            "elapsed_sec": round(self.evacuation_frames_counter / self.fps, 1),
            "valid_detection_count": len(valid_dets),
            "cri_score": round(current_cri_score, 1)
        }

# =========================================================================
# 단체 모듈 자체 시뮬레이션 테스트
# =========================================================================
if __name__ == "__main__":
    print("[TEST] AI 위험 감지 및 Validation Engine (오탐 방지) 테스트 시작...\n")
    
    # 3초 알람 지연(10 FPS = 30프레임), 임계값 70점 설정
    validator = RiskValidator(conf_threshold=0.5, evac_score_threshold=70.0, alarm_delay_sec=3.0)

    # 1. 1프레임 노이즈 튐 시뮬레이션 (75pt가 1번 순간 튈 때)
    print("--- 1. 노이즈 튐 테스트 (75pt 1프레임 발생) ---")
    result = validator.update_and_validate(75.0)
    print(f"상태: {result['status_label']} | 출동여부: {result['is_dispatch_active']}")

    # 2. 다시 정상으로 돌아왔을 때
    result = validator.update_and_validate(20.0)
    print(f"상태: {result['status_label']} | 카운터: {result['evacuation_frames_counter']}\n")

    # 3. 위험 상황 지속 발생 (75pt가 3.5초간 지속 시)
    print("--- 2. 지속 위험 발생 테스트 (75pt가 35프레임 지속) ---")
    for frame in range(1, 36):
        res = validator.update_and_validate(75.0)
        if frame in [1, 15, 30, 35]:
            print(f"프레임 {frame:02d} ({res['elapsed_sec']}s): {res['status_label']} | 출동: {res['is_dispatch_active']}")

    print("\n[SUCCESS] Validation Engine 테스트 정상 완료!")
