"""
train.py
========
Vòng lặp huấn luyện cho ISLR với:
  - CrossEntropyLoss
  - Adam optimizer + CosineAnnealingLR
  - Checkpoint tốt nhất theo val accuracy
  - Logging ra file CSV + console

Cách dùng:
  python train.py
  python train.py --epochs 100 --lr 1e-3 --batch_size 8
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ISLRDataset, build_label_map, KEYPOINTS_ROOT
from model import KeypointLSTM, KeypointLSTMAttention

# ═══════════════════════════════════════════════════════════
# Cấu hình mặc định
# ═══════════════════════════════════════════════════════════
CHECKPOINT_DIR = Path(r"D:/CSLR/ISLR_Project/checkpoints")
LOG_DIR        = Path(r"D:/CSLR/ISLR_Project/logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ISLR — LSTM or LSTM+Attention")
    parser.add_argument("--model",       type=str,   default="lstm",
                        choices=["lstm", "lstm_attn"],
                        help="Kiến trúc model: 'lstm' (mặc định) hoặc 'lstm_attn' (LSTM+Attention)")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--attn_dim",    type=int,   default=128,
                        help="Chiều Attention Head (chỉ dùng khi --model lstm_attn)")
    parser.add_argument("--num_frames",  type=int,   default=15)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Đường dẫn checkpoint để tiếp tục train")
    parser.add_argument("--run_name",    type=str,   default="run",
                        help="Tên run để phân biệt checkpoint/log")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# Train 1 epoch
# ═══════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)                  # (B, num_classes)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc


# ═══════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Directories ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir = CHECKPOINT_DIR / args.run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Label map ──
    label_map = build_label_map("train", KEYPOINTS_ROOT)
    num_classes = len(label_map)
    print(f"[INFO] Classes ({num_classes}): {label_map}")

    # Lưu label_map ra file để dùng lúc inference
    label_map_path = run_ckpt_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Label map đã lưu: {label_map_path}")

    # ── Dataset & DataLoader ──
    train_ds = ISLRDataset(
        split="train",
        label_map=label_map,
        num_frames=args.num_frames,
        augment=True,
    )
    test_ds = ISLRDataset(
        split="test",
        label_map=label_map,
        num_frames=args.num_frames,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,    # Windows: giữ 0 để tránh lỗi multiprocessing
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──
    if args.model == "lstm_attn":
        model = KeypointLSTMAttention(
            num_classes=num_classes,
            input_dim=225,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=True,
            attn_dim=args.attn_dim,
        ).to(device)
        print(f"[INFO] Model: KeypointLSTMAttention (attn_dim={args.attn_dim})")
    else:
        model = KeypointLSTM(
            num_classes=num_classes,
            input_dim=225,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=True,
        ).to(device)
        print(f"[INFO] Model: KeypointLSTM")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Tổng tham số: {total_params:,}")

    # ── Loss / Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    # ── Resume checkpoint ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[INFO] Resume từ epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # ── CSV Log ──
    log_path = LOG_DIR / f"{args.run_name}.csv"
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if start_epoch == 0:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"  Bắt đầu train | {args.epochs} epochs | run_name={args.run_name}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Console log
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # CSV log
        csv_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                              f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.2e}"])
        log_file.flush()

        # ── Checkpoint tốt nhất ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = run_ckpt_dir / "best.pth"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc":    best_val_acc,
                "label_map":       label_map,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ★ Best model lưu → {ckpt_path}  (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1

        # ── Checkpoint định kỳ (mỗi 10 epoch) ──
        if (epoch + 1) % 10 == 0:
            periodic_path = run_ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc":    best_val_acc,
                "label_map":       label_map,
                "args":            vars(args),
            }, periodic_path)

        # ── Early Stopping ──
        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping tại epoch {epoch+1} (patience={args.patience})")
            break

    log_file.close()
    print(f"\n[DONE] Best val_acc = {best_val_acc:.4f}")
    print(f"[DONE] Checkpoint lưu tại: {run_ckpt_dir / 'best.pth'}")
    print(f"[DONE] Log CSV: {log_path}")


if __name__ == "__main__":
    main()
