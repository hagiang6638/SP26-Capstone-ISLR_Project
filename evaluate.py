import argparse
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from dataset import ISLRDataset, FEATURE_DIM
from model import KeypointLSTM

def main():
    parser = argparse.ArgumentParser(description="Evaluate ISLR model and print classification report")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth (VD: checkpoints/exp1/best.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # 1. Load checkpoint
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Không tìm thấy file checkpoint: {args.checkpoint}")
        return
        
    ckpt = torch.load(args.checkpoint, map_location=device)
    label_map = ckpt["label_map"]
    args_saved = ckpt.get("args", {})
    
    # 2. Init model
    model = KeypointLSTM(
        num_classes=len(label_map),
        input_dim=FEATURE_DIM,
        hidden_dim=args_saved.get("hidden_dim", 256),
        num_layers=args_saved.get("num_layers", 2),
        dropout=0.0,
        bidirectional=True
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # 3. Load test dataset
    print(f"[INFO] Đang load tập test dataset...")
    try:
        test_ds = ISLRDataset(split="test", label_map=label_map, augment=False)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if len(test_ds) == 0:
        print("[ERROR] Tập test trống. Không có mẫu nào để đánh giá.")
        return

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 4. Chạy inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    # 5. In báo cáo
    # Tạo danh sách target_names theo đúng thứ tự index 0, 1, 2...
    idx_to_label = {v: k for k, v in label_map.items()}
    target_names = [idx_to_label[i] for i in range(len(label_map))]
    
    print("\n" + "="*60)
    print("               CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)
    
    print("\n" + "="*60)
    print("                 CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    
    # In confusion matrix dạng bảng text
    header = f"{'':>10} | " + " | ".join([f"{name:>8}" for name in target_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(target_names):
        row_str = " | ".join([f"{val:>8}" for val in cm[i]])
        print(f"{name:>10} | {row_str}")
        
if __name__ == "__main__":
    main()
