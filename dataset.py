"""
dataset.py
==========
PyTorch Dataset cho ISLR (Isolated Sign Language Recognition).

Cấu trúc thư mục keypoints:
  D:/CSLR/ISLR_Project/keypoints/{train|test}/{LABEL}/{SAMPLE_ID}.npy
  Mỗi file .npy: shape (T, 75, 3) — T frames, 75 keypoints, (x, y, z)

Xử lý trong __getitem__:
  1. Load file .npy  → (T, 75, 3)
  2. Flatten mỗi frame → (T, 225)
  3. Uniform Sampling / Padding → (NUM_FRAMES, 225)
  4. Normalize (trừ mean, chia std từ training set, hoặc dùng clip-level norm)
  5. Optional: Data Augmentation (jitter, horizontal flip, time warp)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ═══════════════════════════════════════════════════════════
# Cấu hình
# ═══════════════════════════════════════════════════════════
KEYPOINTS_ROOT = Path(r"D:/CSLR/ISLR_Project/keypoints")
NUM_FRAMES     = 15    # số frame cố định sau sampling
NUM_KEYPOINTS  = 75
NUM_COORDS     = 3     # x, y, z
FEATURE_DIM    = NUM_KEYPOINTS * NUM_COORDS  # 225


# ═══════════════════════════════════════════════════════════
# Uniform Frame Sampling
# ═══════════════════════════════════════════════════════════
def uniform_sample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    seq: (T, D) — chuỗi có T frames
    Trả về: (target_len, D) bằng cách lấy mẫu đều hoặc pad cuối
    """
    T = len(seq)
    if T == target_len:
        return seq
    if T > target_len:
        # Lấy target_len frame phân bố đều
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return seq[indices]
    else:
        # Pad bằng cách lặp frame cuối
        pad_len = target_len - T
        padding = np.repeat(seq[-1:], pad_len, axis=0)
        return np.concatenate([seq, padding], axis=0)


# ═══════════════════════════════════════════════════════════
# Augmentation trên tọa độ keypoints
# ═══════════════════════════════════════════════════════════
def augment_keypoints(seq: np.ndarray, config: dict) -> np.ndarray:
    """
    seq: (T, 225) — đã flatten
    config: dict với các key: jitter_std, flip_prob, time_warp_prob
    """
    T, D = seq.shape

    # 1. Coordinate Jitter (thêm nhiễu nhỏ)
    if config.get("jitter_std", 0) > 0:
        noise = np.random.normal(0, config["jitter_std"], size=seq.shape).astype(np.float32)
        seq = seq + noise

    # 2. Horizontal Flip (đổi tay phải/trái)
    # Trong vector phẳng (75, 3): x ở vị trí [i*3], flip x = 1 - x
    # Chỉ flip chiều x, giữ y và z
    if random.random() < config.get("flip_prob", 0):
        seq = seq.reshape(T, NUM_KEYPOINTS, NUM_COORDS).copy()
        seq[:, :, 0] = 1.0 - seq[:, :, 0]  # x = 1 - x
        seq = seq.reshape(T, D)

    # 3. Time Warp nhẹ (giãn/co ngẫu nhiên một đoạn)
    if random.random() < config.get("time_warp_prob", 0):
        # Chỉ thay đổi tốc độ lấy mẫu ±20%
        factor = random.uniform(0.8, 1.2)
        new_len = max(2, int(T * factor))
        indices = np.linspace(0, T - 1, new_len, dtype=float)
        indices = np.clip(np.round(indices).astype(int), 0, T - 1)
        seq = seq[indices]
        seq = uniform_sample(seq, T)  # đưa về độ dài ban đầu

    return seq.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# Dataset Class
# ═══════════════════════════════════════════════════════════
class ISLRDataset(Dataset):
    def __init__(
        self,
        split: str,                     # "train" hoặc "test"
        label_map: Dict[str, int],      # {"TOI": 0, "BAN": 1, "THCH": 2, ...}
        num_frames: int = NUM_FRAMES,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        keypoints_root: Path = KEYPOINTS_ROOT,
    ):
        self.split = split
        self.label_map = label_map
        self.num_frames = num_frames
        self.augment = augment
        self.augment_config = augment_config or {
            "jitter_std": 0.005,
            "flip_prob": 0.3,
            "time_warp_prob": 0.3,
        }
        self.keypoints_root = keypoints_root

        self.samples: List[Tuple[Path, int]] = []  # (path_to_npy, label_idx)
        self._load_samples()

    def _load_samples(self):
        split_dir = self.keypoints_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục keypoints: {split_dir}\n"
                f"Hãy chạy extract_keypoints.py trước!"
            )
        for label, idx in self.label_map.items():
            label_dir = split_dir / label
            if not label_dir.exists():
                print(f"[WARN] Không tìm thấy thư mục label: {label_dir}")
                continue
            for npy_path in sorted(label_dir.glob("*.npy")):
                self.samples.append((npy_path, idx))

        print(f"[Dataset] split={self.split} | {len(self.samples)} samples | "
              f"{len(self.label_map)} classes: {list(self.label_map.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label_idx = self.samples[idx]

        # Load keypoints (T, 75, 3)
        kp = np.load(str(npy_path)).astype(np.float32)
        T = kp.shape[0]

        # Flatten → (T, 225)
        kp_flat = kp.reshape(T, FEATURE_DIM)

        # ── Temporal Sampling ──
        if self.augment and self.split == "train" and T > self.num_frames:
            # Temporal Augmentation: lấy ngẫu nhiên num_frames frame liên tiếp
            start = np.random.randint(0, T - self.num_frames + 1)
            kp_sampled = kp_flat[start : start + self.num_frames]
        else:
            # Test hoặc video ngắn hơn num_frames: uniform_sample (pad/sample đều)
            kp_sampled = uniform_sample(kp_flat, self.num_frames)

        # Augmentation bổ sung (jitter, flip, time_warp — chỉ khi train)
        if self.augment and self.split == "train":
            kp_sampled = augment_keypoints(kp_sampled, self.augment_config)

        # Chuyển sang Tensor
        x = torch.from_numpy(kp_sampled)   # (num_frames, 225)
        y = torch.tensor(label_idx, dtype=torch.long)

        return x, y


# ═══════════════════════════════════════════════════════════
# Helper: tự động xây dựng label_map từ thư mục
# ═══════════════════════════════════════════════════════════
def build_label_map(split: str = "train", keypoints_root: Path = KEYPOINTS_ROOT) -> Dict[str, int]:
    """Tự động tạo label_map theo thứ tự alphabet."""
    split_dir = keypoints_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Chưa có thư mục keypoints: {split_dir}")
    labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    return {label: i for i, label in enumerate(labels)}


# ═══════════════════════════════════════════════════════════
# Test nhanh
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        label_map = build_label_map("train")
        print(f"Label map: {label_map}")

        train_ds = ISLRDataset("train", label_map, augment=True)
        test_ds  = ISLRDataset("test",  label_map, augment=False)

        x, y = train_ds[0]
        print(f"Sample shape: x={x.shape}, y={y}, dtype={x.dtype}")
        print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")
    except FileNotFoundError as e:
        print(e)
