"""
extract_keypoints.py
====================
Đọc dataset từ:
  D:/CSLR/VCSL_dataset/ISLR/raw_video/{train|test}/{LABEL}/{SAMPLE_ID}/{frame_id}.png

Trích xuất 79 keypoints / frame bằng MediaPipe:
  - 26 Face landmarks  (FACE_26_IDX)
  -  7 Body pose       (BODY_POSE_IDX: nose + shoulders + elbows + wrists)
  - 21 Left hand
  - 21 Right hand
  → Tổng: 26 + 7 + 21 + 21 = 75... thực tế script này làm 26 + 7 + 21 + 21 = 75
    nhưng để nhất quán với face_keypoints_realtime.py đặt tên là 79 keypoints
    (mặc định tương đương cách đặt index trong file tham chiếu).

Mỗi sample được lưu dưới dạng:
  D:/CSLR/ISLR_Project/keypoints/{train|test}/{LABEL}/{SAMPLE_ID}.npy
  shape: (num_frames, 75, 3)  — (x, y, z) theo tọa độ normalized [0,1]
  Nếu keypoint không phát hiện → (0.0, 0.0, 0.0)

Cách dùng:
  python extract_keypoints.py
  python extract_keypoints.py --split train
  python extract_keypoints.py --split test
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ═══════════════════════════════════════════════════════════
# Cấu hình đường dẫn
# ═══════════════════════════════════════════════════════════
RAW_VIDEO_ROOT = Path(r"D:\CSLR\VCSL_dataset\15fps\train_model")
OUTPUT_ROOT    = Path(r"D:\CSLR\ISLR_Project\keypoints")
MP_MODEL_DIR   = Path(__file__).parent / ".mp_models"

# ═══════════════════════════════════════════════════════════
# Keypoint indices (đồng bộ với face_keypoints_realtime.py)
# ═══════════════════════════════════════════════════════════
FACE_26_IDX = [
    2, 4,
    185, 39, 37, 0, 267, 269, 409,
    181, 17, 405,
    130, 173,
    398, 359,
    63, 66,
    296, 293,
    127, 215, 149,
    356, 435, 378,
]  # 26 điểm

BODY_POSE_IDX = [0, 11, 13, 15, 12, 14, 16]  # nose + L(shoulder,elbow,wrist) + R(shoulder,elbow,wrist) = 7 điểm

# Tổng: 26 face + 7 pose + 21 left_hand + 21 right_hand = 75
NUM_KEYPOINTS = 75   # 26 + 7 + 21 + 21

# ═══════════════════════════════════════════════════════════
# Version detection
# ═══════════════════════════════════════════════════════════
_mp_ver = tuple(int(x) for x in mp.__version__.split(".")[:2])
MP_NEW_API = _mp_ver >= (0, 10)
print(f"[INFO] MediaPipe {mp.__version__} → {'Tasks API (>=0.10)' if MP_NEW_API else 'Legacy API (<0.10)'}")


# ═══════════════════════════════════════════════════════════
# Download models (Tasks API)
# ═══════════════════════════════════════════════════════════
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
            print(f"[INFO] Downloading {fname} ...")
            urllib.request.urlretrieve(url, dst)
            print(f"[INFO] Saved → {dst}")
    return model_dir


# ═══════════════════════════════════════════════════════════
# Khởi tạo estimators
# ═══════════════════════════════════════════════════════════
def build_estimators():
    if MP_NEW_API:
        from mediapipe.tasks import python as _mp_tasks_python
        from mediapipe.tasks.python import vision as _mp_tasks_vision

        model_dir = download_mp_models(MP_MODEL_DIR)
        RunningMode = _mp_tasks_vision.RunningMode

        pose_est = _mp_tasks_vision.PoseLandmarker.create_from_options(
            _mp_tasks_vision.PoseLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / "pose_landmarker.task")
                ),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        hand_est = _mp_tasks_vision.HandLandmarker.create_from_options(
            _mp_tasks_vision.HandLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / "hand_landmarker.task")
                ),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        face_est = _mp_tasks_vision.FaceLandmarker.create_from_options(
            _mp_tasks_vision.FaceLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / "face_landmarker.task")
                ),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
    else:
        pose_est = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        hand_est = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        face_est = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    return pose_est, hand_est, face_est


# ═══════════════════════════════════════════════════════════
# Trích xuất keypoints từ 1 frame (ảnh BGR numpy array)
# Trả về numpy (75, 3) — không resize ảnh gốc
# ═══════════════════════════════════════════════════════════
def extract_frame_keypoints(bgr_frame, pose_est, hand_est, face_est, timestamp_ms: int = 0):
    kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    img_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    ptr = 0  # con trỏ vào kp

    if MP_NEW_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # ── Face (26 + 4 = 30 điểm) ──
        face_res = face_est.detect(mp_image)
        all_face_idx = FACE_26_IDX
        if face_res.face_landmarks:
            lms = face_res.face_landmarks[0]
            for idx in all_face_idx:
                if idx < len(lms):
                    lm = lms[idx]
                    kp[ptr] = [lm.x, lm.y, lm.z]
                ptr += 1
        else:
            ptr += len(all_face_idx)  # bỏ qua 30 slot

        # ── Body Pose (7 điểm) ──
        pose_res = pose_est.detect(mp_image)
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            for idx in BODY_POSE_IDX:
                if idx < len(lms):
                    lm = lms[idx]
                    kp[ptr] = [lm.x, lm.y, lm.z]
                ptr += 1
        else:
            ptr += len(BODY_POSE_IDX)

        # ── Hands (21 left + 21 right = 42 điểm) ──
        hand_res = hand_est.detect(mp_image)
        left_hand  = None
        right_hand = None
        if hand_res.hand_landmarks and hand_res.handedness:
            for lms, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = hd[0].category_name  # "Left" / "Right"
                if label == "Left":
                    left_hand = lms
                else:
                    right_hand = lms

        for hand_lms in [left_hand, right_hand]:
            if hand_lms is not None:
                for lm in hand_lms:
                    kp[ptr] = [lm.x, lm.y, lm.z]
                    ptr += 1
            else:
                ptr += 21  # zero-padding nếu không thấy tay

    else:
        # ── Legacy API ──
        # Face
        face_res = face_est.process(img_rgb)
        all_face_idx = FACE_26_IDX
        if face_res.multi_face_landmarks:
            for idx in all_face_idx:
                lm = face_res.multi_face_landmarks[0].landmark[idx]
                kp[ptr] = [lm.x, lm.y]
                ptr += 1
        else:
            ptr += len(all_face_idx)

        # Pose
        pose_res = pose_est.process(img_rgb)
        if pose_res.pose_landmarks:
            for idx in BODY_POSE_IDX:
                lm = pose_res.pose_landmarks.landmark[idx]
                kp[ptr] = [lm.x, lm.y]
                ptr += 1
        else:
            ptr += len(BODY_POSE_IDX)

        # Hands (Legacy: "Right" = camera right = tay TRÁI người ký)
        hand_res = hand_est.process(img_rgb)
        left_hand  = None
        right_hand = None
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for hand_lms, handedness in zip(
                hand_res.multi_hand_landmarks, hand_res.multi_handedness
            ):
                label = handedness.classification[0].label
                # Legacy ngược: "Right" camera = Left signer
                if label == "Right":
                    left_hand = hand_lms.landmark
                else:
                    right_hand = hand_lms.landmark

        for hand_lms in [left_hand, right_hand]:
            if hand_lms is not None:
                for lm in hand_lms:
                    kp[ptr] = [lm.x, lm.y, lm.z]
                    ptr += 1
            else:
                ptr += 21

    return kp


# ═══════════════════════════════════════════════════════════
# Xử lý 1 sample (thư mục chứa các frame PNG)
# Trả về numpy (T, 75, 3) với T = số frame
# ═══════════════════════════════════════════════════════════
def process_sample(sample_dir: Path, pose_est, hand_est, face_est):
    frame_files = sorted(sample_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(sample_dir.glob("*.jpg"))
    if not frame_files:
        print(f"  [WARN] Không tìm thấy frame nào trong {sample_dir}")
        return None

    keypoints_seq = []
    for f in frame_files:
        bgr = cv2.imread(str(f))  # đọc theo size gốc, không resize
        if bgr is None:
            print(f"  [WARN] Không đọc được {f}")
            kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        else:
            kp = extract_frame_keypoints(bgr, pose_est, hand_est, face_est)
        keypoints_seq.append(kp)

    return np.stack(keypoints_seq, axis=0).astype(np.float32)  # (T, 75, 3)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main(splits):
    pose_est, hand_est, face_est = build_estimators()

    total_samples = 0
    failed_samples = 0

    for split in splits:
        split_dir = RAW_VIDEO_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] Không tìm thấy split: {split_dir}")
            continue

        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name

            for sample_dir in sorted(label_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                sample_id = sample_dir.name
                out_dir   = OUTPUT_ROOT / split / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path  = out_dir / f"{sample_id}.npy"

                if out_path.exists():
                    print(f"  [SKIP] {split}/{label}/{sample_id}.npy đã tồn tại")
                    continue

                print(f"  [PROC] {split}/{label}/{sample_id} ...", end=" ", flush=True)
                t0 = time.time()

                result = process_sample(sample_dir, pose_est, hand_est, face_est)
                if result is not None:
                    np.save(str(out_path), result)
                    elapsed = time.time() - t0
                    print(f"shape={result.shape}  ({elapsed:.1f}s)")
                    total_samples += 1
                else:
                    failed_samples += 1
                    print("FAILED")

    print(f"\n[DONE] Tổng: {total_samples} samples thành công, {failed_samples} thất bại.")

    # Đóng estimators (Tasks API)
    if MP_NEW_API:
        pose_est.close()
        hand_est.close()
        face_est.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất MediaPipe keypoints cho ISLR dataset")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Split cần xử lý (default: all)",
    )
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    main(splits)
