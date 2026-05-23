# ISLR Project — Nhận diện Ngôn ngữ Ký hiệu Đơn lẻ (Isolated SLR)

Dự án nhận diện ngôn ngữ ký hiệu từng từ một, sử dụng MediaPipe để trích xuất 75 keypoints và mô hình Bi-LSTM nhẹ.

---

## Cấu trúc thư mục

```
D:/CSLR/ISLR_Project/
├── extract_keypoints.py    ← Bước 1: Trích xuất keypoints từ dataset ảnh
├── dataset.py              ← PyTorch Dataset + Augmentation
├── model.py                ← KeypointLSTM model
├── train.py                ← Vòng lặp huấn luyện
├── predict_realtime.py     ← Demo webcam real-time
├── requirements.txt
├── keypoints/              ← Output của extract_keypoints.py
│   ├── train/{LABEL}/{SAMPLE}.npy
│   └── test/{LABEL}/{SAMPLE}.npy
├── checkpoints/            ← Model checkpoints
│   └── {run_name}/
│       ├── best.pth
│       ├── epoch_010.pth
│       └── label_map.json
└── logs/
    └── {run_name}.csv      ← Training log
```

Dataset nguồn (chỉ đọc, không sửa):
```
D:/CSLR/VCSL_dataset/ISLR/raw_video/
├── train/{LABEL}/{SAMPLE_ID}/*.png
└── test/{LABEL}/{SAMPLE_ID}/*.png
```

---

## 75 Keypoints

| Nhóm | Số điểm | Mô tả |
|------|---------|-------|
| Face | 26 | Các điểm mặt đại diện (FACE_26_IDX) |
| Body Pose | 7 | Mũi + vai + khuỷu + cổ tay (trái & phải) |
| Left Hand | 21 | 21 điểm bàn tay trái người ký |
| Right Hand | 21 | 21 điểm bàn tay phải người ký |
| **Tổng** | **75** | |

Mỗi keypoint có 3 giá trị `(x, y, z)` → Feature vector mỗi frame = **225 chiều**.

---

## Quy trình chạy

### Bước 1 — Trích xuất keypoints

```bash
cd D:/CSLR/ISLR_Project
python extract_keypoints.py --split all
# Hoặc từng split:
python extract_keypoints.py --split train
python extract_keypoints.py --split test
```

Output: `keypoints/train/TOI/TOI-0001.npy` với shape `(T, 75, 3)`

### Bước 2 — Huấn luyện

```bash
python train.py --run_name exp1 --epochs 100 --batch_size 8 --lr 1e-3
```

Tham số chính:
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--epochs` | 100 | Số epoch tối đa |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate ban đầu |
| `--num_frames` | 30 | Số frame cố định sau sampling |
| `--hidden_dim` | 256 | Số unit ẩn LSTM |
| `--num_layers` | 2 | Số lớp LSTM |
| `--dropout` | 0.3 | Dropout |
| `--patience` | 20 | Early stopping patience |
| `--resume` | None | Tiếp tục từ checkpoint |
| `--run_name` | "run" | Tên run |

### Bước 3 — Demo real-time

```bash
python predict_realtime.py --checkpoint checkpoints/exp1/best.pth
```

- Nhấn **SPACE** → bắt đầu ghi 30 frame → model dự đoán → hiển thị kết quả
- Nhấn **Q hoặc ESC** → thoát

---

## Kiến trúc Model (KeypointLSTM)

```
Input (B, 30, 225)
    ↓
Linear(225→256) + BatchNorm + ReLU + Dropout
    ↓
Bi-LSTM(256, 2 layers)
    ↓
Concat hidden states [forward, backward] → (B, 512)
    ↓
Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→num_classes)
    ↓
Logits (B, num_classes)
```

Tham số: ~1.5M (rất nhẹ, phù hợp dataset nhỏ)

---

## Checkpoint Format

```python
{
    "epoch": int,
    "model_state": OrderedDict,
    "optimizer_state": OrderedDict,
    "scheduler_state": OrderedDict,
    "best_val_acc": float,
    "label_map": {"TOI": 0, "BAN": 1, ...},
    "args": {...}
}
```

Resume training:
```bash
python train.py --resume checkpoints/exp1/best.pth --run_name exp1_cont
```
