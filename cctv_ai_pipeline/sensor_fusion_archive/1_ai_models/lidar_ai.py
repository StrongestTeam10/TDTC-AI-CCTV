# lidar_ai.py: LiDAR 3D 포인트 클라우드 데이터를 가우시안 원형 라벨링 기반의 BEV 격자 히트맵 데이터셋으로 로드하고 
# 학습시키는 딥러닝 트레이닝 스크립트입니다.
import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# =========================================================================
# 0. 재현성 + A100 최적화 세팅
# =========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =========================================================================
# 1. 데이터셋 정의 (가우시안 원형 라벨링 + 증강)
# =========================================================================
class SeongsanLidarDatasetGaussian(Dataset):
    def __init__(self, data_dir, split='training', max_points=8192, grid_size=0.25,
                 x_range=(-15, 15), y_range=(-15, 25), z_range=(-1.0, 2.5),
                 augment=False):
        self.max_points = max_points
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.augment = augment
        self.nx = int((x_range[1] - x_range[0]) / grid_size)
        self.ny = int((y_range[1] - y_range[0]) / grid_size)

        bin_files = sorted(glob.glob(os.path.join(data_dir, split, 'unzipped_lidar', '**/*.bin'), recursive=True))
        label_files = sorted(glob.glob(os.path.join(data_dir, split, 'label', '**/*.json'), recursive=True))

        self.samples = []
        skipped = 0

        for b_file in bin_files:
            b_name = os.path.splitext(os.path.basename(b_file))[0]
            matched_json = [j for j in label_files if b_name in os.path.basename(j)]
            if matched_json:
                self.samples.append((b_file, matched_json[0]))
            else:
                skipped += 1

        print(f"📦 [{split.upper()}] 총 {len(self.samples)}개 샘플 성공적으로 로드 완료! (매칭 실패 {skipped}개 스킵)")

    def __len__(self):
        return len(self.samples)

    def draw_gaussian(self, heatmap, center_x, center_y, radius=2, sigma=1.0):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = center_x + dx, center_y + dy
                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    gaussian_val = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    heatmap[ny, nx] = max(heatmap[ny, nx], gaussian_val)

    def __getitem__(self, idx):
        bin_path, json_path = self.samples[idx]

        scan = np.fromfile(bin_path, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :3].copy()

        # Z축 높이 필터링
        z_mask = (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        points = points[z_mask]

        with open(json_path, 'r', encoding='utf-8-sig') as f:
            label_data = json.load(f)

        target_coords = []
        if 'lidar_classes' in label_data:
            for obj in label_data['lidar_classes']:
                cx, cy, _ = obj['3D_Bbox_position']
                target_coords.append([cx, cy])
        target_coords = np.array(target_coords, dtype=np.float32).reshape(-1, 2)

        # 데이터 증강 (train 세트만)
        if self.augment:
            if random.random() < 0.5:
                points[:, 0] *= -1
                if len(target_coords) > 0:
                    target_coords[:, 0] *= -1

            angle = np.random.uniform(-np.pi / 12, np.pi / 12)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            points[:, :2] = points[:, :2] @ rot.T
            if len(target_coords) > 0:
                target_coords = target_coords @ rot.T

            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
            if len(target_coords) > 0:
                target_coords *= scale

        # 포인트 수 정규화
        num_pts = len(points)
        if num_pts >= self.max_points:
            choice = np.random.choice(num_pts, self.max_points, replace=False)
            points = points[choice]
        else:
            if num_pts > 0:
                pad = np.zeros((self.max_points - num_pts, 3), dtype=np.float32)
                points = np.vstack((points, pad))
            else:
                points = np.zeros((self.max_points, 3), dtype=np.float32)

        # 히트맵 생성
        heatmap_target = np.zeros((self.ny, self.nx), dtype=np.float32)
        for cx, cy in target_coords:
            x_idx = int((cx - self.x_range[0]) / self.grid_size)
            y_idx = int((cy - self.y_range[0]) / self.grid_size)
            if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                self.draw_gaussian(heatmap_target, x_idx, y_idx, radius=2)

        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(heatmap_target, dtype=torch.float32).unsqueeze(0),
            target_coords.tolist(),
            os.path.basename(bin_path)
        )

def collate_fn(batch):
    pts, targets, coords, names = zip(*batch)
    return torch.stack(pts), torch.stack(targets), list(coords), list(names)

# =========================================================================
# 2. 손실 함수 (CenterNet Focal Loss)
# =========================================================================
class CenterNetFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super(CenterNetFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        pos_inds = target.eq(1.0).float()
        neg_inds = target.lt(1.0).float()

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

# =========================================================================
# 3. PointPillar + CenterPoint 모델 (dtype 불일치 픽스 적용!)
# =========================================================================
class PointPillarCenterPointModel(nn.Module):
    def __init__(self, x_range=(-15, 15), y_range=(-15, 25), grid_size=0.25):
        super(PointPillarCenterPointModel, self).__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.nx = int((x_range[1] - x_range[0]) / grid_size)
        self.ny = int((y_range[1] - y_range[0]) / grid_size)

        self.pfn = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.center_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_points):
        B, N, _ = x_points.shape
        device = x_points.device

        pts_flat = x_points.reshape(B * N, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)

        valid_mask = (pts_flat[:, 0] != 0) | (pts_flat[:, 1] != 0) | (pts_flat[:, 2] != 0)

        x_idx = ((pts_flat[:, 0] - self.x_range[0]) / self.grid_size).long()
        y_idx = ((pts_flat[:, 1] - self.y_range[0]) / self.grid_size).long()
        in_range = (x_idx >= 0) & (x_idx < self.nx) & (y_idx >= 0) & (y_idx < self.ny)

        mask = valid_mask & in_range
        if mask.sum() == 0:
            return torch.zeros((B, 1, self.ny, self.nx), device=device, dtype=x_points.dtype)

        pts_valid = pts_flat[mask]
        x_idx_v = x_idx[mask]
        y_idx_v = y_idx[mask]
        batch_idx_v = batch_idx[mask]

        pillar_center_x = self.x_range[0] + (x_idx_v.float() + 0.5) * self.grid_size
        pillar_center_y = self.y_range[0] + (y_idx_v.float() + 0.5) * self.grid_size

        xc = pts_valid[:, 0] - pillar_center_x
        yc = pts_valid[:, 1] - pillar_center_y
        zc = pts_valid[:, 2] - pts_valid[:, 2].mean()

        xp = pts_valid[:, 0] - pillar_center_x
        yp = pts_valid[:, 1] - pillar_center_y

        feat_in = torch.stack([
            pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2],
            xc, yc, zc, xp, yp
        ], dim=1)

        if feat_in.size(0) > 1:
            feat = self.pfn(feat_in)
        else:
            self.pfn.eval()
            with torch.no_grad():
                feat = self.pfn(feat_in)
            self.pfn.train()

        flat_pillar_idx = batch_idx_v * (self.ny * self.nx) + y_idx_v * self.nx + x_idx_v

        # 📌 [핵심 수정] dtype=feat.dtype 추가로 bfloat16과 float32 불일치 에러 100% 해결!
        canvas_flat = torch.zeros((B * self.ny * self.nx, 64), device=device, dtype=feat.dtype)
        canvas_flat = canvas_flat.scatter_reduce(
            0,
            flat_pillar_idx.unsqueeze(1).expand(-1, 64),
            feat,
            reduce="amax",
            include_self=True
        )

        canvas = canvas_flat.view(B, self.ny, self.nx, 64).permute(0, 3, 1, 2).contiguous()

        bev_feat = self.backbone(canvas)
        heatmap = self.center_head(bev_feat)
        return heatmap

def calculate_metrics(preds, targets, threshold=0.3):
    pred_binary = (preds >= threshold).float()
    target_binary = (targets >= threshold).float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return acc, f1

# =========================================================================
# 4. 학습 실행 루프
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 사용 디바이스: {device}")
if device.type == "cuda":
    print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

full_dataset = SeongsanLidarDatasetGaussian("/content/data", split='training', augment=True)

val_ratio = 0.1
val_size = max(1, int(len(full_dataset) * val_ratio))
train_size = len(full_dataset) - val_size

train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_dataset_noaug = SeongsanLidarDatasetGaussian("/content/data", split='training', augment=False)
val_subset = torch.utils.data.Subset(val_dataset_noaug, val_subset.indices)

BATCH_SIZE = 16
NUM_WORKERS = 2

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True,
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

model = PointPillarCenterPointModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = CenterNetFocalLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

save_dir = '/content/drive/MyDrive/LiDAR_Weights_Results'
os.makedirs(save_dir, exist_ok=True)
best_ckpt_path = os.path.join(save_dir, 'seongsan_centerpoint_best.pth')
last_ckpt_path = os.path.join(save_dir, 'seongsan_centerpoint_last.pth')

print(f"\n🚀 학습 루프 진입 (batch={BATCH_SIZE}, samples={len(train_subset)})...")
epochs = 30
best_val_f1 = -1.0
patience_counter = 0
early_stop_patience = 8

for epoch in range(1, epochs + 1):
    # --- Train ---
    model.train()
    total_loss, total_acc, total_f1, count = 0.0, 0.0, 0.0, 0

    for step, batch in enumerate(train_loader):
        pts_tensor, target_tensor, _, _ = batch
        pts_tensor = pts_tensor.to(device, non_blocking=True)
        target_tensor = target_tensor.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            preds = model(pts_tensor)
            loss = criterion(preds, target_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        acc, f1 = calculate_metrics(preds.detach().float(), target_tensor)

        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        count += 1

        if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
            print(f"   ㄴ [Train] Batch [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f} | F1: {f1:.4f}")

    avg_loss = total_loss / max(count, 1)
    avg_acc = (total_acc / max(count, 1)) * 100
    avg_f1 = total_f1 / max(count, 1)
    print(f"📈 [Epoch {epoch:02d}/{epochs:02d}] Train Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | F1: {avg_f1:.4f}")

    # --- Validation ---
    model.eval()
    val_loss, val_acc, val_f1, val_count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            pts_tensor, target_tensor, _, _ = batch
            pts_tensor = pts_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                preds = model(pts_tensor)
                loss = criterion(preds, target_tensor)

            acc, f1 = calculate_metrics(preds.float(), target_tensor)
            val_loss += loss.item()
            val_acc += acc
            val_f1 += f1
            val_count += 1

    avg_val_loss = val_loss / max(val_count, 1)
    avg_val_acc = (val_acc / max(val_count, 1)) * 100
    avg_val_f1 = val_f1 / max(val_count, 1)
    print(f"🔍 [Epoch {epoch:02d}/{epochs:02d}] Val   Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2f}% | F1: {avg_val_f1:.4f}\n")

    scheduler.step(avg_val_loss)

    torch.save(model.state_dict(), last_ckpt_path)

    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        patience_counter = 0
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"   ✅ Best 모델 갱신 (Val F1: {best_val_f1:.4f}) -> {best_ckpt_path}")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"⏹️  {early_stop_patience} epoch 동안 개선 없어 조기 종료합니다.")
            break

print(f"\n✅ 학습 완료. Best Val F1: {best_val_f1:.4f}")