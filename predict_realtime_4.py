"""
predict_realtime_4.py  (Auto-Trigger Mode - 3D)
=============================================
Demo nhận diện ngôn ngữ ký hiệu tự động kích hoạt khi có bàn tay.

Luồng hoạt động:
- Nhấn SPACE để bật hệ thống (chuyển sang trạng thái SẴN SÀNG).
- Khi SẴN SÀNG: Camera chạy nhưng KHÔNG vẽ keypoint, KHÔNG dự đoán.
- Khi người dùng đưa tay lên (phát hiện có tay) -> Tự động chuyển sang ĐANG THU.
- ĐANG THU: Vẽ keypoint, thu đủ 15 frames -> Dự đoán -> NGHỈ.
- NGHỈ (1.5s): Chờ người dùng hạ tay xuống. Vẫn vẽ keypoint để tạo độ mượt.
- Sau khi hết 1.5s VÀ rút tay khỏi màn hình -> Quay lại SẴN SÀNG.

Cách dùng:
  python predict_realtime_4.py --checkpoint checkpoints/exp1/best.pth
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from dataset import NUM_FRAMES, NUM_KEYPOINTS, FEATURE_DIM, uniform_sample
from model import KeypointLSTM, KeypointLSTMAttention

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

    model_type = args_saved.get("model", "lstm")
    if model_type == "lstm_attn":
        model = KeypointLSTMAttention(
            num_classes=len(label_map),
            input_dim=FEATURE_DIM,
            hidden_dim=args_saved.get("hidden_dim", 256),
            num_layers=args_saved.get("num_layers", 2),
            dropout=0.0,
            bidirectional=True,
            attn_dim=args_saved.get("attn_dim", 128)
        ).to(device)
    else:
        model = KeypointLSTM(
            num_classes=len(label_map),
            input_dim=FEATURE_DIM,
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
    arr = np.stack(kp_seq, axis=0)                # (T, 75, 3)
    arr = arr.reshape(len(kp_seq), FEATURE_DIM)   # (T, 225)
    arr = uniform_sample(arr, num_frames)          # (15, 225)
    x   = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 15, 225)

    logits = model(x)
    probs  = F.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    conf   = probs[idx].item()
    return idx, conf, probs.cpu().numpy()


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


def draw_hud(frame, sentence, state, progress, fps, show_ui):
    h, w = frame.shape[:2]

    # Xác định màu và chữ dựa trên trạng thái
    if state == "PAUSED":
        color = (0, 0, 255) # Đỏ
        state_txt = "TAM DUNG (Nhan Space de bat)"
    elif state == "WAITING_HAND":
        color = (255, 255, 0) # Cyan/Vàng
        state_txt = "SAN SANG (Hay dua tay len...)"
    elif state == "RECORDING":
        color = (0, 255, 0) # Xanh lá
        state_txt = "DANG THU (Hay ky...)"
    else: # DELAY
        color = (100, 100, 100) # Xám
        state_txt = "NGHI (Ha tay xuong...)"

    # Luôn hiển thị trạng thái TẠM DỪNG để cảnh báo.
    # Còn các trạng thái lúc đang chạy thì sẽ bị ẩn nếu show_ui = False
    if state == "PAUSED" or show_ui:
        cv2.rectangle(frame, (0, 0), (w, h), color, 4)
        cv2.putText(frame, state_txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Progress bar cho RECORDING và DELAY (chỉ ẩn khi nhấn H)
    if show_ui and state not in ["PAUSED", "WAITING_HAND"]:
        bar_h = 10
        bar_y = 55
        bar_w = 200
        cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (15, bar_y), (15 + fill_w, bar_y + bar_h), color, -1)

    # Sentence
    if sentence:
        sentence_str = " ".join(sentence)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
            
        # Tính kích thước chữ
        bbox = font.getbbox(sentence_str)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        tx = (w - text_w) // 2
        ty = h - 60
        
        # Vẽ nền đen
        cv2.rectangle(frame, (tx - 10, ty - 5), (tx + text_w + 10, ty + text_h + 10), (0, 0, 0), -1)
        
        # Vẽ chữ tiếng Việt bằng PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((tx, ty - bbox[1]), sentence_str, font=font, fill=(0, 255, 0)) # fill=(R, G, B)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    help_txt = "Space: Bat/Tat he thong | H: An UI | C: Xoa | Q: Thoat"
    cv2.putText(frame, help_txt, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame


# Từ điển ánh xạ từ folder name (không dấu) sang Tiếng Việt hiển thị (có dấu)
VIETNAMESE_MAP = {
    "AN": "ĂN",
    "BAN": "BẠN",
    "DI": "ĐI",
    "MEO": "MÈO",
    "THICH": "THÍCH",
    "TOI": "TÔI"
}

def main():
    parser = argparse.ArgumentParser(description="ISLR Demo — Auto Trigger Mode (3D)")
    parser.add_argument("--checkpoint",  type=str, required=True, help="Đường dẫn file .pth")
    parser.add_argument("--camera",      type=int,   default=0)
    parser.add_argument("--record_frames", type=int, default=15,
                        help="Số frame thu thập cho mỗi từ (Mặc định: 15)")
    parser.add_argument("--delay_sec",   type=float, default=1,
                        help="Số giây nghỉ giữa các từ để hạ tay (Mặc định: 1.5s)")
    parser.add_argument("--conf_thresh", type=float, default=0.0,
                        help="Ngưỡng tự tin tối thiểu (Mặc định: 0.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    model, idx_to_label = load_model(args.checkpoint, device)
    print(f"[INFO] Labels gốc từ model: {idx_to_label}")

    pose_est, hand_est, face_est = build_estimators_image()
    cap = cv2.VideoCapture(args.camera)

    kp_buffer = []
    sentence = []
    
    # Flags & States
    system_active = False    # Bắt đầu ở trạng thái TẠM DỪNG
    state = "PAUSED"
    delay_end_time = 0.0
    show_ui = True
    progress = 0.0

    fps = 0.0
    prev_time = time.time()

    print(f"\n{'='*50}")
    print(" BẮT ĐẦU CHẾ ĐỘ NHẬN DIỆN KÍCH HOẠT THEO TAY (AUTO-TRIGGER)")
    print(" - Nhấn SPACE để BẬT/TẮT hệ thống")
    print(" - Khi bật, chỉ thu khi CÓ BÀN TAY xuất hiện trong khung hình")
    print(" - Nhấn H: Ẩn/Hiện giao diện")
    print(" - Nhấn C: Xóa câu hiện tại")
    print(" - Nhấn Q hoặc ESC: Thoát")
    print(f"{'='*50}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0: fps = 0.7 * fps + 0.3 * (1.0 / dt)

        # STATE MACHINE LOGIC
        if not system_active:
            state = "PAUSED"
            progress = 0.0
            # KHÔNG trích xuất hay vẽ keypoint khi TẠM DỪNG để tiết kiệm CPU/GPU
        else:
            # LUÔN LUÔN trích xuất keypoint khi hệ thống ĐÃ BẬT
            kp = extract_frame_kp(frame, pose_est, hand_est, face_est)
            
            # Kiểm tra xem có bàn tay trong khung hình không
            # Tọa độ bàn tay nằm từ index 33 đến 74.
            # Nếu tọa độ x > 0 tức là Mediapipe nhận diện được ít nhất 1 điểm của bàn tay.
            has_hand = np.any(kp[33:75, 0] > 0)

            if state == "WAITING_HAND":
                progress = 0.0
                if has_hand:
                    # Bàn tay xuất hiện -> Bắt đầu thu
                    state = "RECORDING"
                    kp_buffer = [kp]
                    draw_keypoints(frame, kp)
                else:
                    # Không có tay -> Không vẽ keypoint, màn hình trống
                    pass

            elif state == "RECORDING":
                kp_buffer.append(kp)
                draw_keypoints(frame, kp)
                
                progress = len(kp_buffer) / args.record_frames
                
                if len(kp_buffer) >= args.record_frames:
                    # Đã thu đủ -> Dự đoán
                    idx, conf, _ = predict(kp_buffer, model, device, args.record_frames)
                    label_goc = idx_to_label[idx]
                    
                    # Ánh xạ sang Tiếng Việt có dấu
                    label_vi = VIETNAMESE_MAP.get(label_goc, label_goc)
                    
                    print(f"[PREDICT] {label_vi} (Conf: {conf:.2f})")
                    
                    if conf >= args.conf_thresh:
                        sentence.append(label_vi)
                        
                    # Chuyển sang Delay tự động
                    state = "DELAY"
                    delay_end_time = time.time() + args.delay_sec

            elif state == "DELAY":
                progress = max(0, (delay_end_time - curr_time) / args.delay_sec)
                # Trong lúc hạ tay xuống, VẪN VẼ keypoint để demo trông liên tục
                draw_keypoints(frame, kp)
                
                # Chuyển về trạng thái chờ KHI: Đã hết thời gian 1.5s VÀ Đã hạ tay khỏi khung hình
                if curr_time >= delay_end_time and not has_hand:
                    state = "WAITING_HAND"

        frame = draw_hud(frame, sentence, state, progress, fps, show_ui)

        cv2.imshow("ISLR Auto Trigger Demo (3D)", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q") or key == 27: 
            break
        elif key == ord("c"):
            sentence.clear()
            print("[INFO] Đã xóa câu.")
        elif key == ord("h"):
            show_ui = not show_ui
            print(f"[INFO] Hiển thị UI: {show_ui}")
        elif key == 32:  # Phím SPACE
            system_active = not system_active
            if system_active:
                print("[INFO] HỆ THỐNG BẬT - Đang chờ bàn tay xuất hiện.")
                state = "WAITING_HAND"
                kp_buffer.clear()
            else:
                print("[INFO] HỆ THỐNG TẠM DỪNG.")

    cap.release()
    cv2.destroyAllWindows()
    if MP_NEW_API:
        pose_est.close()
        hand_est.close()
        face_est.close()

if __name__ == "__main__":
    main()
