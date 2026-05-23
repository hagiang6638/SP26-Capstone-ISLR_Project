"""
predict_realtime_2.py  (Continuous Mode - 3D)
=============================================
Demo nhận diện ngôn ngữ ký hiệu LIÊN TỤC qua webcam — không cần nhấn phím.
Sử dụng điểm 3D (x, y, z) cho ISLR_Project.

Cách dùng:
  python predict_realtime_2.py --checkpoint checkpoints/exp1/best.pth
"""

import argparse
import collections
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
            )
        )
        hand_est = _v.HandLandmarker.create_from_options(
            _v.HandLandmarkerOptions(
                base_options=_p.BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
                running_mode=RunMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.3,
            )
        )
        face_est = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_p.BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
                running_mode=RunMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.3,
            )
        )
    else:
        pose_est = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
        hand_est = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
        face_est = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    return pose_est, hand_est, face_est


def extract_frame_kp(bgr_frame, pose_est, hand_est, face_est):
    """Trích xuất 75 keypoints từ 1 frame. (3D: x, y, z)"""
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
                if hd.classification[0].label == "Right":
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


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_map: dict = ckpt["label_map"]
    args_saved = ckpt.get("args", {})

    model = KeypointLSTM(
        num_classes=len(label_map),
        input_dim=FEATURE_DIM,  # 225
        hidden_dim=args_saved.get("hidden_dim", 256),
        num_layers=args_saved.get("num_layers", 2),
        dropout=0.0,
        bidirectional=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label


@torch.no_grad()
def predict(kp_seq, model, device, num_frames=NUM_FRAMES):
    """
    kp_seq: list of (75, 3) numpy arrays
    """
    arr = np.stack(kp_seq, axis=0)                # (T, 75, 3)
    arr = arr.reshape(len(kp_seq), FEATURE_DIM)   # (T, 225)
    arr = uniform_sample(arr, num_frames)          # (30, 225)
    x   = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 30, 225)

    logits = model(x)
    probs  = F.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    conf   = probs[idx].item()
    return idx, conf, probs.cpu().numpy()


def compute_movement_energy(kp_buffer, hand_start=33, hand_end=75):
    if len(kp_buffer) < 2:
        return 0.0

    total_energy = 0.0
    for i in range(1, len(kp_buffer)):
        prev_hands = kp_buffer[i - 1][hand_start:hand_end]
        curr_hands = kp_buffer[i][hand_start:hand_end]
        diff = curr_hands - prev_hands
        energy = np.sqrt(np.sum(diff ** 2, axis=1)).sum()
        total_energy += energy

    return total_energy / (len(kp_buffer) - 1)


def draw_keypoints(frame, kp):
    h, w = frame.shape[:2]
    for i in range(26):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)
    for i in range(26, 33):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 255), -1)
    for i in range(33, 54):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 0, 0), -1)
    for i in range(54, 75):
        x, y = kp[i, 0], kp[i, 1]
        if x > 0 or y > 0:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)
    return frame


def draw_hud(frame, sentence, current_label, current_conf,
             energy, energy_thresh, conf_thresh, buffer_fill):
    h, w = frame.shape[:2]

    bar_h = 8
    bar_y = h - bar_h - 5
    bar_w = int(w * 0.4)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * min(buffer_fill, 1.0))
    color_bar = (0, 200, 0) if buffer_fill >= 1.0 else (0, 140, 200)
    cv2.rectangle(frame, (10, bar_y), (10 + fill_w, bar_y + bar_h), color_bar, -1)
    cv2.putText(frame, f"Buffer", (10, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    energy_str = f"Energy: {energy:.4f}"
    e_color = (0, 255, 0) if energy >= energy_thresh else (100, 100, 100)
    cv2.putText(frame, energy_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, e_color, 1)

    if current_label:
        pred_color = (0, 255, 255) if current_conf >= conf_thresh else (100, 100, 100)
        cv2.putText(frame, f"Raw: {current_label} ({current_conf:.0%})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

    if sentence:
        sentence_str = " ".join(sentence)
        text_size = cv2.getTextSize(sentence_str, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = h - 50
        cv2.rectangle(frame, (tx - 10, ty - text_size[1] - 10), (tx + text_size[0] + 10, ty + 10), (0, 0, 0), -1)
        cv2.putText(frame, sentence_str, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(frame, "C: Xoa cau | Q: Thoat", (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="ISLR Demo — Nhận diện liên tục (3D)")
    parser.add_argument("--checkpoint",     type=str, required=True, help="Đường dẫn file .pth")
    parser.add_argument("--camera",         type=int,   default=0)
    parser.add_argument("--window_size",    type=int,   default=15,
                        help="Kích thước cửa sổ trượt (default: 30 frames)")
    parser.add_argument("--step_size",      type=int,   default=3,
                        help="Số frame giữa mỗi lần dự đoán")
    parser.add_argument("--conf_thresh",    type=float, default=0.65,
                        help="Ngưỡng confidence tối thiểu")
    parser.add_argument("--stability",      type=int,   default=3,
                        help="Số lần dự đoán liên tiếp giống nhau để chấp nhận")
    parser.add_argument("--cooldown",       type=int,   default=10,
                        help="Số frame chờ sau khi nhận diện 1 gloss")
    parser.add_argument("--energy_thresh",  type=float, default=0.02,
                        help="Ngưỡng năng lượng chuyển động tối thiểu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    model, idx_to_label = load_model(args.checkpoint, device)
    print(f"[INFO] Labels: {idx_to_label}")

    pose_est, hand_est, face_est = build_estimators_image()
    cap = cv2.VideoCapture(args.camera)

    kp_buffer       = collections.deque(maxlen=args.window_size)
    frame_count     = 0
    recent_preds    = collections.deque(maxlen=args.stability)
    cooldown_remain = 0
    sentence        = []
    last_accepted   = None

    current_raw_label = None
    current_raw_conf  = 0.0

    fps = 0.0
    prev_time = time.time()

    print(f"\n[INFO] Bắt đầu nhận diện liên tục. Q/ESC để thoát. C để xóa câu.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        kp = extract_frame_kp(frame, pose_est, hand_est, face_est)
        kp_buffer.append(kp)
        frame_count += 1
        draw_keypoints(frame, kp)

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0: fps = 0.7 * fps + 0.3 * (1.0 / dt)

        if cooldown_remain > 0:
            cooldown_remain -= 1

        buffer_fill = len(kp_buffer) / args.window_size
        if len(kp_buffer) >= args.window_size and frame_count >= args.step_size:
            frame_count = 0
            energy = compute_movement_energy(list(kp_buffer))

            if energy >= args.energy_thresh:
                idx, conf, _ = predict(list(kp_buffer), model, device, args.window_size)
                label = idx_to_label[idx]
                current_raw_label = label
                current_raw_conf  = conf

                print(f"[DEBUG] Dự đoán thô: {label} ({conf:.2%})")

                if conf >= args.conf_thresh:
                    recent_preds.append(label)
                else:
                    recent_preds.append(None)
            else:
                current_raw_label = "(nghi)"
                current_raw_conf  = 0.0
                recent_preds.append(None)

            if len(recent_preds) >= args.stability and cooldown_remain <= 0:
                last_n = list(recent_preds)[-args.stability:]
                if all(p is not None for p in last_n) and len(set(last_n)) == 1:
                    accepted_label = last_n[0]
                    if accepted_label != last_accepted:
                        sentence.append(accepted_label)
                        last_accepted   = accepted_label
                        cooldown_remain = args.cooldown
                        print(f"  ✓ Nhận diện: {accepted_label}  |  Câu: {' '.join(sentence)}")
                        recent_preds.clear()

        draw_hud(frame, sentence, current_raw_label, current_raw_conf,
                 compute_movement_energy(list(kp_buffer)) if len(kp_buffer) >= 2 else 0.0,
                 args.energy_thresh, args.conf_thresh, buffer_fill)

        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("ISLR Continuous Demo (3D) — Q: Thoat | C: Xoa cau", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27: break
        elif key == ord("c"):
            sentence.clear()
            last_accepted = None
            recent_preds.clear()
            print("[INFO] Đã xóa câu.")

    cap.release()
    cv2.destroyAllWindows()
    if MP_NEW_API:
        pose_est.close()
        hand_est.close()
        face_est.close()

if __name__ == "__main__":
    main()
