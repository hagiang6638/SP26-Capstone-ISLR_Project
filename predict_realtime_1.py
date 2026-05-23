"""
predict_realtime.py
===================
Demo nhận diện ngôn ngữ ký hiệu real-time qua webcam.

Luồng:
  1. Nhấn SPACE (hoặc nút trên màn hình) → bắt đầu ghi NUM_FRAMES frame
  2. Trích xuất keypoints từng frame bằng MediaPipe
  3. Đưa qua model KeypointLSTM → ra nhãn và độ tin cậy
  4. Hiển thị kết quả trên màn hình

Cách dùng:
  python predict_realtime.py --checkpoint D:/CSLR/ISLR_Project/checkpoints/run/best.pth
  python predict_realtime.py --checkpoint <path> --camera 0
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from dataset import NUM_FRAMES, NUM_KEYPOINTS, FEATURE_DIM, uniform_sample
from model import KeypointLSTM

MP_MODEL_DIR = Path(__file__).parent / ".mp_models"

# ═══════════════════════════════════════════════════════════
# MediaPipe setup (giống extract_keypoints.py)
# ═══════════════════════════════════════════════════════════
FACE_26_IDX   = [2,4,185,39,37,0,267,269,409,181,17,405,130,173,398,359,63,66,296,293,127,215,149,356,435,378]
BODY_POSE_IDX  = [0, 11, 13, 15, 12, 14, 16]

_mp_ver = tuple(int(x) for x in mp.__version__.split(".")[:2])
MP_NEW_API = _mp_ver >= (0, 10)


def download_mp_models(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "pose_landmarker.task": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
            "/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        ),
        "hand_landmarker.task": (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker"
            "/hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        "face_landmarker.task": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker"
            "/face_landmarker/float16/latest/face_landmarker.task"
        ),
    }
    for fname, url in models.items():
        dst = model_dir / fname
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
    return model_dir


def build_estimators_image():
    """Estimators ở chế độ IMAGE (không cần timestamp)."""
    if MP_NEW_API:
        from mediapipe.tasks import python as _p
        from mediapipe.tasks.python import vision as _v
        model_dir = download_mp_models(MP_MODEL_DIR)
        RunMode = _v.RunningMode

        pose_est = _v.PoseLandmarker.create_from_options(
            _v.PoseLandmarkerOptions(
                base_options=_p.BaseOptions(model_asset_path=str(model_dir / "pose_landmarker.task")),
                running_mode=RunMode.IMAGE, num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        hand_est = _v.HandLandmarker.create_from_options(
            _v.HandLandmarkerOptions(
                base_options=_p.BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
                running_mode=RunMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        face_est = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_p.BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
                running_mode=RunMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
    else:
        pose_est = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
        hand_est = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
        face_est = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    return pose_est, hand_est, face_est


def extract_frame_kp(bgr_frame, pose_est, hand_est, face_est):
    """Trích xuất 75 keypoints từ 1 frame."""
    kp  = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    ptr = 0

    if MP_NEW_API:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        face_res = face_est.detect(mp_img)
        for idx in FACE_26_IDX:
            if face_res.face_landmarks and idx < len(face_res.face_landmarks[0]):
                lm = face_res.face_landmarks[0][idx]
                kp[ptr] = [lm.x, lm.y, lm.z]
            ptr += 1

        pose_res = pose_est.detect(mp_img)
        for idx in BODY_POSE_IDX:
            if pose_res.pose_landmarks and idx < len(pose_res.pose_landmarks[0]):
                lm = pose_res.pose_landmarks[0][idx]
                kp[ptr] = [lm.x, lm.y, lm.z]
            ptr += 1

        hand_res = hand_est.detect(mp_img)
        left_h, right_h = None, None
        if hand_res.hand_landmarks and hand_res.handedness:
            for lms, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
                if hd[0].category_name == "Left":
                    left_h = lms
                else:
                    right_h = lms
        for h in [left_h, right_h]:
            if h:
                for lm in h:
                    kp[ptr] = [lm.x, lm.y, lm.z]
                    ptr += 1
            else:
                ptr += 21
    else:
        face_res = face_est.process(img)
        for idx in FACE_26_IDX:
            if face_res.multi_face_landmarks:
                lm = face_res.multi_face_landmarks[0].landmark[idx]
                kp[ptr] = [lm.x, lm.y, lm.z]
            ptr += 1

        pose_res = pose_est.process(img)
        for idx in BODY_POSE_IDX:
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark[idx]
                kp[ptr] = [lm.x, lm.y, lm.z]
            ptr += 1

        hand_res = hand_est.process(img)
        left_h, right_h = None, None
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for h, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                if hd.classification[0].label == "Right":  # camera right = signer left
                    left_h = h.landmark
                else:
                    right_h = h.landmark
        for h in [left_h, right_h]:
            if h:
                for lm in h:
                    kp[ptr] = [lm.x, lm.y, lm.z]
                    ptr += 1
            else:
                ptr += 21

    return kp  # (75, 3)


# ═══════════════════════════════════════════════════════════
# Load model từ checkpoint
# ═══════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_map: dict = ckpt["label_map"]
    args_saved = ckpt.get("args", {})

    model = KeypointLSTM(
        num_classes=len(label_map),
        input_dim=FEATURE_DIM,
        hidden_dim=args_saved.get("hidden_dim", 256),
        num_layers=args_saved.get("num_layers", 2),
        dropout=0.0,   # tắt dropout khi inference
        bidirectional=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Đảo ngược label_map → {idx: label}
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label


# ═══════════════════════════════════════════════════════════
# Dự đoán từ chuỗi keypoints
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def predict(kp_seq: list, model, device, num_frames=NUM_FRAMES):
    """
    kp_seq: list of (75, 3) numpy arrays
    Trả về: (label_idx, confidence)
    """
    arr = np.stack(kp_seq, axis=0)                # (T, 75, 3)
    arr = arr.reshape(len(kp_seq), FEATURE_DIM)   # (T, 225)
    arr = uniform_sample(arr, num_frames)          # (30, 225)
    x   = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 30, 225)

    logits = model(x)                              # (1, num_classes)
    probs  = F.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    conf   = probs[idx].item()
    return idx, conf


# ═══════════════════════════════════════════════════════════
# Vẽ keypoints lên frame
# ═══════════════════════════════════════════════════════════
def draw_keypoints(frame, kp):
    """kp: (75, 3), vẽ điểm lên frame theo pixel."""
    h, w = frame.shape[:2]
    # Face (26 điểm): xanh lá
    for i in range(26):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)
    # Pose (7 điểm): vàng
    for i in range(26, 33):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 255), -1)
    # Left hand (21): xanh dương
    for i in range(33, 54):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 0, 0), -1)
    # Right hand (21): đỏ
    for i in range(54, 75):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)
    return frame


# ═══════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn file .pth")
    parser.add_argument("--camera",     type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--conf_thresh",type=float, default=0.5,
                        help="Ngưỡng confidence để hiển thị kết quả")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Đang load model từ {args.checkpoint} ...")

    model, idx_to_label = load_model(args.checkpoint, device)
    print(f"[INFO] Labels: {idx_to_label}")

    pose_est, hand_est, face_est = build_estimators_image()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {args.camera}")

    # Trạng thái
    STATE_IDLE      = "idle"
    STATE_RECORDING = "recording"
    state        = STATE_IDLE
    kp_buffer    = []
    result_label = None
    result_conf  = 0.0
    result_time  = 0.0
    RESULT_SHOW_SEC = 3.0   # Hiển thị kết quả bao nhiêu giây

    print("\n[INFO] Nhấn SPACE để bắt đầu ghi | Q để thoát")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = extract_frame_kp(frame, pose_est, hand_est, face_est)
        draw_keypoints(frame, kp)

        # ── State machine ──
        if state == STATE_RECORDING:
            kp_buffer.append(kp)
            progress = len(kp_buffer) / args.num_frames
            # Thanh tiến trình
            bar_w = int(frame.shape[1] * 0.6)
            cv2.rectangle(frame, (20, frame.shape[0] - 30), (20 + bar_w, frame.shape[0] - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, frame.shape[0] - 30), (20 + int(bar_w * progress), frame.shape[0] - 10), (0, 200, 0), -1)
            cv2.putText(frame, f"Dang ghi... {len(kp_buffer)}/{args.num_frames}",
                        (20, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(kp_buffer) >= args.num_frames:
                # Dự đoán
                idx, conf = predict(kp_buffer, model, device, args.num_frames)
                result_label = idx_to_label[idx]
                result_conf  = conf
                result_time  = time.time()
                kp_buffer    = []
                state        = STATE_IDLE
                print(f"  → Dự đoán: {result_label}  (conf={conf:.2%})")

        # ── Hiển thị kết quả ──
        if result_label and (time.time() - result_time) < RESULT_SHOW_SEC:
            color = (0, 255, 0) if result_conf >= args.conf_thresh else (0, 165, 255)
            cv2.putText(frame, f"{result_label}  ({result_conf:.0%})",
                        (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
        elif result_label and (time.time() - result_time) >= RESULT_SHOW_SEC:
            result_label = None

        # ── HUD ──
        status_txt = "GHI" if state == STATE_RECORDING else "Nhan SPACE de ghi"
        cv2.putText(frame, status_txt, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if state == STATE_RECORDING else (180, 180, 180), 2)

        cv2.imshow("ISLR Demo — SPACE: Ghi | Q: Thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if state == STATE_IDLE:
                state     = STATE_RECORDING
                kp_buffer = []
                print(f"[INFO] Bắt đầu ghi {args.num_frames} frames ...")

    cap.release()
    cv2.destroyAllWindows()
    if MP_NEW_API:
        pose_est.close()
        hand_est.close()
        face_est.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
