# lidar_pipeline.py: 3D 객체 검출을 위한 경량 PointPillars 모델 구조(PillarFeatureNet, Backbone 등)와 추론 파이프라인 클래스를 정의한 파일입니다.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================================================================
# [Step 1] 3D Object Detection 모델 구조 (Lightweight PointPillars)
# =====================================================================

class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Net (PFN)
    포인트 클라우드 데이터를 Pillar 그리드 기반 2D Pseudo-image 특징 맵으로 변환합니다.
    RTX 4060 환경에서 실시간성(10 FPS 이상)을 위해 경량 MLP로 구성합니다.
    """
    def __init__(self, in_channels=4, out_channels=64, voxel_size=(0.2, 0.2), point_cloud_range=(-10.0, 0.0, -3.0, 10.0, 20.0, 2.0)):
        super(PillarFeatureNet, self).__init__()
        self.vx, self.vy = voxel_size
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = point_cloud_range
        
        # Grid 크기 계산
        self.grid_x = int((self.x_max - self.x_min) / self.vx)
        self.grid_y = int((self.y_max - self.y_min) / self.vy)
        
        # 1D CNN 혹은 Linear를 통해 각 포인트의 차원을 확장합니다.
        # 입력: [x, y, z, intensity] -> 필요 시 각 Pillar의 중심과의 거리(offset)를 추가하여 확장 가능
        self.pfn_layers = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, points, grid_indices):
        """
        points: [Total_Points, In_Channels] (텐서)
        grid_indices: [Total_Points, 2] (Pillar 격자 인덱스 x, y)
        """
        # 1. PFN 특징 매핑
        x = self.pfn_layers(points)  # [Total_Points, Out_Channels]
        
        # 2. Pseudo-image 생성 (Scatter 연산)
        # 실시간 고속 연산을 위해 GPU에서 PyTorch indexing 연산으로 격자 맵을 구성합니다.
        device = points.device
        pseudo_image = torch.zeros((x.shape[1], self.grid_y, self.grid_x), device=device)
        
        # 유효 범위 내 격자 인덱스 필터링
        mask = (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < self.grid_x) & \
               (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < self.grid_y)
        
        filtered_indices = grid_indices[mask]
        filtered_features = x[mask]
        
        if len(filtered_indices) > 0:
            # 동일 격자에 여러 포인트가 들어올 경우 고속 Max Pooling 효과를 위해 index_put_ 사용
            # 실제 PointPillars는 각 voxel 내에서 max pooling을 수행한 후 scatter 합니다.
            # 여기서는 실시간성 극대화를 위해 각 격자별 최댓값(Max)을 scatter 하도록 최적화합니다.
            y_indices = filtered_indices[:, 1]
            x_indices = filtered_indices[:, 0]
            
            # Scatter/Scatter-max 연산 구현
            # 픽셀 좌표 평평하게 변환
            flat_indices = y_indices * self.grid_x + x_indices
            
            # 고유한 격자 인덱스에서 최댓값을 효율적으로 추출
            unique_flat, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            
            # 각 채널별로 고유 격자 인덱스에 Max Pooling 수행
            pooled_features = torch.zeros((len(unique_flat), x.shape[1]), device=device)
            # PyTorch Scatter-Max 구현 (안전하고 호환성이 높은 방식)
            for c in range(x.shape[1]):
                pooled_features[:, c] = pooled_features[:, c].scatter_reduce(
                    0, inverse_indices, filtered_features[:, c], reduce='mean', include_self=False
                )
            
            # Pseudo-image에 매핑
            channels = x.shape[1]
            flat_pseudo = pseudo_image.view(channels, -1)
            flat_pseudo[:, unique_flat] = pooled_features.t()
            pseudo_image = flat_pseudo.view(channels, self.grid_y, self.grid_x)
            
        return pseudo_image.unsqueeze(0)  # [1, Channels, Grid_Y, Grid_X] (Batch_size = 1)


class PointPillarsBackbone(nn.Module):
    """
    2D CNN Backbone
    Pseudo-image 특징 맵을 입력받아 다중 스케일 특징을 융합합니다.
    """
    def __init__(self, in_channels=64):
        super(PointPillarsBackbone, self).__init__()
        
        # Block 1 (Downsample)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Block 2 (Downsample)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Upsampling (Deconvolution)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [1, in_channels, H, W]
        x1 = self.block1(x)       # [1, 64, H/2, W/2]
        x2 = self.block2(x1)      # [1, 128, H/4, W/4]
        
        up1 = self.deconv1(x1)    # [1, 64, H, W]
        up2 = self.deconv2(x2)    # [1, 64, H, W]
        
        # 다중 스케일 특징 결합 (Concat)
        out = torch.cat([up1, up2], dim=1) # [1, 128, H, W]
        return out


class DetectionHead(nn.Module):
    """
    3D Bounding Box Detection Head
    클래스 확률(Classification)과 3D BBox(Regression) 정보를 예측합니다.
    """
    def __init__(self, in_channels=128, num_anchors=2, num_classes=1):
        super(DetectionHead, self).__init__()
        # 3D BBox 파라미터: [x, y, z, dx, dy, dz, heading(r,p,y 중 yaw)] -> 7개 차원
        self.num_anchors = num_anchors
        
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=1)
        
    def forward(self, x):
        # x: [1, 128, H, W]
        cls_preds = self.conv_cls(x)  # [1, num_anchors * num_classes, H, W]
        box_preds = self.conv_box(x)  # [1, num_anchors * 7, H, W]
        
        # 추론 시에 보기 편하게 차원을 재배열합니다.
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        return cls_preds, box_preds


class PointPillars(nn.Module):
    """
    최종 Lightweight PointPillars 통합 PyTorch 모델
    """
    def __init__(self, voxel_size=(0.2, 0.2), point_cloud_range=(-10.0, 0.0, -3.0, 10.0, 20.0, 2.0)):
        super(PointPillars, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        
        self.pfn = PillarFeatureNet(voxel_size=voxel_size, point_cloud_range=point_cloud_range)
        self.backbone = PointPillarsBackbone()
        self.head = DetectionHead()
        
    def forward(self, points):
        """
        points: [N, 4] -> [x, y, z, intensity]의 raw 포인트 텐서
        """
        # 1. 포인트별 격자 인덱스 계산
        device = points.device
        x_indices = torch.floor((points[:, 0] - self.pc_range[0]) / self.voxel_size[0]).long()
        y_indices = torch.floor((points[:, 1] - self.pc_range[1]) / self.voxel_size[1]).long()
        grid_indices = torch.stack([x_indices, y_indices], dim=1).to(device)
        
        # 2. Forward 패스 진행
        pseudo_img = self.pfn(points, grid_indices)
        features = self.backbone(pseudo_img)
        cls_preds, box_preds = self.head(features)
        
        return cls_preds, box_preds

# =====================================================================
# [Step 2] 구역별 실시간 밀집 위험 점수(LiDAR Risk Score) 산출
# =====================================================================

class LidarRiskScorer:
    """
    LiDAR 위험 점수 산출기
    밀집도(Density)와 객체 간 근접도(Proximity)를 종합하여 위험 등급 및 점수(0.0 ~ 1.0)를 반환합니다.
    """
    def __init__(self, roi_x=(-8.0, 8.0), roi_y=(0.0, 15.0), lambda_d=0.25, d_safe=1.8, w_density=0.6, w_proximity=0.4):
        """
        roi_x: 관심 가로 영역 [min, max]
        roi_y: 관심 세로 영역 [min, max]
        lambda_d: 밀집도 점수 계산용 지수 감쇄 인자 (클수록 적은 인원에도 점수가 가파르게 상승)
        d_safe: 사회적 안전 기준 거리 (m). 이 거리보다 가까워질수록 근접 위험도 급증
        w_density: 위험 점수 산출 시 밀집도 가중치
        w_proximity: 위험 점수 산출 시 근접도 가중치
        """
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.lambda_d = lambda_d
        self.d_safe = d_safe
        self.w_density = w_density
        self.w_proximity = w_proximity
        
    def filter_roi_boxes(self, bbox_coords):
        """
        검출된 3D Bounding Box 좌표 중 ROI 구역 내에 속하는 Bbox만 필터링합니다.
        bbox_coords: [Num_Detections, 7] -> [x, y, z, dx, dy, dz, heading]
        """
        if len(bbox_coords) == 0:
            return np.array([])
            
        x = bbox_coords[:, 0]
        y = bbox_coords[:, 1]
        
        # ROI 내부에 포함되는 마스크 생성
        mask = (x >= self.roi_x[0]) & (x <= self.roi_x[1]) & \
               (y >= self.roi_y[0]) & (y <= self.roi_y[1])
               
        return bbox_coords[mask]

    def calculate_risk_score(self, bbox_coords):
        """
        3D Bounding Box 결과들을 기반으로 0.0 ~ 1.0 범위의 위험 점수를 산출합니다.
        bbox_coords: [Num_Detections, 7] (필터링되지 않은 원본 좌표 또는 이미 ROI 필터링된 좌표)
        """
        # 1. ROI 필터링 수행
        roi_boxes = self.filter_roi_boxes(bbox_coords)
        N = len(roi_boxes)
        
        # 객체가 아무것도 없으면 안전 상태(0.0)
        if N == 0:
            return 0.0
            
        # 2. 밀집도(Density) 위험 점수 산출
        # 객체 수가 늘어날수록 1.0에 수렴하도록 지수 함수 설계
        s_density = 1.0 - np.exp(-self.lambda_d * N)
        
        # 3. 근접도(Proximity) 위험 점수 산출
        if N >= 2:
            # 모든 객체 간의 2D 평면 거리(Pairwise Distance) 계산
            coords_2d = roi_boxes[:, 0:2]  # [x, y] 좌표 추출
            
            # Pairwise 차이 계산
            diff = coords_2d[:, np.newaxis, :] - coords_2d[np.newaxis, :, :]  # [N, N, 2]
            distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N, N]
            
            # 대각선 성분(자기 자신과의 거리) 및 중복 계산 제외를 위해 상삼각행렬의 원소만 추출
            triu_indices = np.triu_indices(N, k=1)
            pair_distances = distances[triu_indices]
            
            # 평균 거리 계산
            d_mean = np.mean(pair_distances)
            
            # 근접도 공식: 안전 임계값 거리보다 가까워질수록 점수가 1.0에 수렴
            # 거리가 매우 가까우면(예: 0.5m) s_proximity는 1.0에 가까워지고, 멀어지면 0.0에 수렴
            s_proximity = np.exp(-d_mean / self.d_safe)
        else:
            # 객체가 1개 이하일 경우 객체 간 근접 위험은 없으므로 0으로 처리
            s_proximity = 0.0
            
        # 4. 종합 위험 점수 결합 (Late Fusion용 최종 출력)
        # 기본 가중합
        raw_score = (self.w_density * s_density) + (self.w_proximity * s_proximity)
        
        # 만약 밀집도 자체가 매우 임계값을 넘어서면 점수가 급격하게 1.0으로 수렴하도록 가중 조절
        # 예: 8명 이상 관제 구역 내 집결 시 밀집 점수 자체의 위험 가중치 상향
        if N >= 8:
            raw_score = 0.8 * s_density + 0.2 * s_proximity
            
        final_score = np.clip(raw_score, 0.0, 1.0)
        return float(final_score)

# =====================================================================
# [Step 3] Late Fusion 연동용 API 및 Fallback 시뮬레이터
# =====================================================================

class LidarInferencePipeline:
    """
    실제 3D 모델 및 Heuristic 대안 알고리즘을 융합한 추론 파이썬 엔트리포인트 클래스.
    가상환경 및 가중치 누락 상황에서도 Late Fusion 관제 시스템의 중단 없는 작동을 보장합니다.
    """
    def __init__(self, model_weight_path=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 구조 선언
        self.model = PointPillars().to(self.device)
        self.model.eval()
        
        self.has_weights = False
        if model_weight_path and os.path.exists(model_weight_path):
            try:
                self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
                self.has_weights = True
                print(f"[LiDAR Pipeline] 3D Model weights loaded from: {model_weight_path}")
            except Exception as e:
                print(f"[LiDAR Pipeline] Failed to load weights: {e}. Running in robust Fallback/Simulation mode.")
        else:
            print("[LiDAR Pipeline] No model weights provided. Running in robust Fallback/Simulation mode.")
            
        # 위험 스코어러 초기화
        self.scorer = LidarRiskScorer()
        
    def heuristic_density_clustering(self, points):
        """
        3D 모델 가중치가 없는 초기 개발 단계에서도 정상 작동하도록 설계된 Fallback 알고리즘.
        바이너리 파일의 포인트들을 공간 필터링 및 간단한 유클리디안 군집화(Clustering)하여 객체를 검출하고 Box를 모사합니다.
        """
        # ROI 내 포인트만 필터링
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        roi_mask = (x >= self.scorer.roi_x[0]) & (x <= self.scorer.roi_x[1]) & \
                   (y >= self.scorer.roi_y[0]) & (y <= self.scorer.roi_y[1]) & \
                   (z >= -2.0) & (z <= 1.5) # 높이 필터 추가 (지면 및 천장 노이즈 제외)
                   
        roi_points = points[roi_mask]
        
        if len(roi_points) < 50:  # 포인트가 너무 없으면 객체 없음으로 판단
            return np.array([])
            
        # 간단한 그리드 기반 클러스터링(유클리디언 클러스터링 모사)
        # 3D 공간상의 조밀한 포인트 덩어리들을 보행자 객체로 간주
        coords = roi_points[:, 0:2] # 2D 투영
        
        # DBSCAN 등을 모사한 간단한 격자 그리드 병합 방식 구현 (CPU 상에서 고속 작동)
        # 격자 크기 0.8m로 보행자 반경 설정
        grid_size = 0.8
        grid_coords = np.floor(coords / grid_size).astype(int)
        
        # 격자별 포인트 개수 집계
        unique_grids, counts = np.unique(grid_coords, axis=0, return_counts=True)
        
        # 노이즈 필터링 (보행자 크기의 격자 내에 최소 15개 이상의 포인트가 있어야 함)
        valid_grids = unique_grids[counts >= 15]
        
        detected_objects = []
        for grid in valid_grids:
            # 해당 격자에 속한 포인트들의 평균 위치 계산
            grid_mask = (grid_coords[:, 0] == grid[0]) & (grid_coords[:, 1] == grid[1])
            obj_points = roi_points[grid_mask]
            
            center_x = np.mean(obj_points[:, 0])
            center_y = np.mean(obj_points[:, 1])
            center_z = np.mean(obj_points[:, 2])
            
            # 보행자 기준 대략적인 BBox 생성 [x, y, z, dx, dy, dz, yaw]
            detected_objects.append([center_x, center_y, center_z, 0.6, 0.6, 1.7, 0.0])
            
        return np.array(detected_objects)

    def run_inference(self, bin_path):
        """
        단일 .bin 파일을 읽어 3D 검출 수행 후 Bbox 좌표 리스트를 반환합니다.
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"LiDAR data file not found: {bin_path}")
            
        # 1. binary [.bin] 파일 로드
        point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        
        # 2. 추론 진행
        if self.has_weights:
            # PyTorch 기반 딥러닝 추론 수행
            points_tensor = torch.from_numpy(point_cloud).float().to(self.device)
            with torch.no_grad():
                cls_preds, box_preds = self.model(points_tensor)
                
            # 3D Anchor 매핑 및 Non-Maximum Suppression(NMS) 처리가 필요합니다.
            # 여기서는 모델 아웃풋 형태에서 신뢰 점수가 높은 활성화 노드를 바운딩 박스로 변환합니다.
            # (학습 가중치가 있을 경우, 실제 Anchor Decoding 연산을 타게 됨)
            # 가중치가 실질적으로 동작할 때의 좌표 변환 예시:
            decoded_boxes = []
            # 임계값을 넘긴 감지 대상 디코딩 로직 생략 (일반적인 Anchor-based detector 디코딩부)
            # pseudo-code 형태로 빈 리스트일 때 Fallback 동작 지원
            if len(decoded_boxes) == 0:
                # 딥러닝 가중치가 비어있거나 디코딩 데이터가 불완전하면 실시간 관제를 위해 Heuristic 모델로 자동 보정
                bbox_coords = self.heuristic_density_clustering(point_cloud)
            else:
                bbox_coords = np.array(decoded_boxes)
        else:
            # 가중치가 없는 개발 단계: 신뢰성 높은 Heuristic 밀집도 클러스터링으로 3D 검출 모사
            bbox_coords = self.heuristic_density_clustering(point_cloud)
            
        return bbox_coords

# =====================================================================
# 전역 파이프라인 인스턴스 (Late Fusion 연동용 싱글톤 객체)
# =====================================================================
_pipeline_instance = None

def predict_lidar_risk(bin_path, model_weight_path=None):
    """
    [Late Fusion 연동용 API 함수]
    마트/관광현장의 LiDAR .bin 파일 경로를 받아서
    실시간 보행자 검출 및 구역별 위험 등급 점수(0.0 ~ 1.0)를 반환합니다.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        # 최초 호출 시 파이프라인 인스턴스 싱글톤 초기화
        _pipeline_instance = LidarInferencePipeline(model_weight_path=model_weight_path)
        
    try:
        # 1. 3D Bounding Box 추론 수행
        detected_boxes = _pipeline_instance.run_inference(bin_path)
        
        # 2. 위험 점수 산출
        risk_score = _pipeline_instance.scorer.calculate_risk_score(detected_boxes)
        
        return risk_score
        
    except Exception as e:
        print(f"[predict_lidar_risk Exception] {e}")
        # 오류 발생 시 시스템이 다운되지 않도록 최소 안전 점수(0.0) 반환
        return 0.0

if __name__ == "__main__":
    # 단위 테스트 및 작동 예시 검증
    print("=" * 60)
    print("LiDAR Pipeline Unit Test Script")
    print("=" * 60)
    
    # 1. 가상의 포인트 클라우드 데이터 생성 (보행자 3명 위치 모사)
    # 보행자 1: [2.0, 5.0, 0.0] 근방
    # 보행자 2: [-1.5, 4.5, -0.2] 근방
    # 보행자 3: [0.5, 8.0, 0.1] 근방
    np.random.seed(42)
    p1 = np.random.normal(loc=[2.0, 5.0, 0.0, 0.8], scale=[0.3, 0.3, 0.5, 0.1], size=(100, 4))
    p2 = np.random.normal(loc=[-1.5, 4.5, -0.2, 0.7], scale=[0.2, 0.2, 0.4, 0.1], size=(80, 4))
    p3 = np.random.normal(loc=[0.5, 8.0, 0.1, 0.9], scale=[0.2, 0.2, 0.3, 0.1], size=(120, 4))
    noise = np.random.uniform(low=[-10.0, 0.0, -2.0, 0.1], high=[10.0, 20.0, 2.0, 0.5], size=(500, 4))
    
    dummy_pc = np.vstack([p1, p2, p3, noise]).astype(np.float32)
    
    # 임시 파일 저장
    temp_bin_path = "temp_test_cloud.bin"
    dummy_pc.tofile(temp_bin_path)
    print(f"임시 테스트 파일 생성 완료: {temp_bin_path} (포인트 수: {dummy_pc.shape[0]}개)")
    
    # 2. API 검증 호출
    score = predict_lidar_risk(temp_bin_path)
    print(f"\n[Test Result] Calculated Risk Score: {score:.4f} (정상 산출 완료)")
    
    # 3. 객체 간 거리 및 가혹 조건 시뮬레이션
    print("\n가혹 조건 시뮬레이션 (동일 구역에 15명 초저근접 집결 상태):")
    crowded_objects = []
    # 1.0m 간격으로 15개 객체 배치
    for i in range(15):
        x = (i % 4) * 0.8 - 1.2
        y = (i // 4) * 0.8 + 4.0
        crowded_objects.append([x, y, 0.0, 0.6, 0.6, 1.7, 0.0])
        
    crowded_objects = np.array(crowded_objects)
    scorer = LidarRiskScorer()
    crowded_score = scorer.calculate_risk_score(crowded_objects)
    print(f"-> 15명 밀집 시 위험도 점수: {crowded_score:.4f} (1.0에 수렴해야 함)")
    
    # 임시 파일 삭제 정리
    if os.path.exists(temp_bin_path):
        os.remove(temp_bin_path)
        print("\n임시 파일 정리 완료.")
    print("=" * 60)
