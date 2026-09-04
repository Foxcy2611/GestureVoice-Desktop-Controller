"""
====================
Tên script: Eval_Gesture_Webcam.py
Tác dụng: Chạy mô hình gesture trên webcam và hiển thị nhãn/confidence real-time.
====================
"""

"""
Phase 4 - GestureVoice Desktop Controller
Test model gesture (Gestures_MLP_Model.keras) qua webcam thoi gian thuc.

QUAN TRỌNG: Mọi thứ đều phải trùng khớp với Script: Collect_Gesture_Data.py với mọi tham số
đã được cài đặt

vvv Done
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np 
import cv2
import mediapipe as mp
from keras import models

# ========= LINK =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "P3_Train_Model", "Output_Gesture", "Gesture_MLP_Model.keras")
Z_PATH = os.path.join(BASE_DIR, "P2_Data_Preprocessing", "Output_npy", "Z_classes.npy")

NUM_LANDMARK = 21
CONFIDENCE_THRESHOLD = 0.7  # Chỉ hiện thị nhãn nếu >= ngưỡng này


def Load_Model_And_Classes():
    model = models.load_model(MODEL_PATH)
    classes = np.load(Z_PATH, allow_pickle=True)

    return model, classes


def Normalize_Landmarks(hand_landmarks):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )  # (21, 3)

    wrist = coords[0].copy()
    coords_translated = coords - wrist  

    max_dist = np.max(np.linalg.norm(coords_translated, axis=1))
    if max_dist >= 1e-6:
        coords_translated = coords_translated / max_dist  

    return coords_translated.flatten()  # (63,) [x0, y0, z0, ...]


def Predict_Gesture(model, classes, feature_vector):
    # Reshape về dạng 1x63 phù hợp vs input đầu vào mạng
    # (1, -1) vs -1 cho tự tính số cột
    X = feature_vector.reshape(1, -1)  # (1, 63)
    # output dự đoán dạng  1x10
    probs = model.predict(X, verbose=0)[0]  # (10,)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    label = str(classes[class_id])
    return label, confidence # return về nhãn + sự tự tin


def main():
    print("Đang tải mô hình và bảng nhãn...")
    model, classes = Load_Model_And_Classes()
    print(f"  + Mô hình: {MODEL_PATH}")
    print(f"  + Các lớp: {list(classes)}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không mở được webcam.")
        return

    print("\nĐang chạy real-time. Nhấn 'q' để thoát.")
    print("Gợi ý test: thay đổi khoảng cách / góc nghiêng / ánh sáng KHÁC lúc thu dữ liệu.")
    print("Lưu ý: tay trái CHỈ có 2 trường hợp đã học (one, two) - giơ trường hợp khác")
    print("       bằng tay trái sẽ bị mô hình gán nhầm vào 1 trong 10 nhãn đã biết.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Lay ca handedness (Left/Right) tu MediaPipe song song voi du doan
            # cua model - dung de doi chieu bang mat: neu MediaPipe thay "Left"
            # nhung model lai du doan "right_xxx", do la dau hieu model dang
            # nham lan - rat huu ich khi demo/bao ve de giai thich ket qua.
            handedness_list = results.multi_handedness

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, handedness_list
            ):
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

                mp_hand_label = handedness.classification[0].label  # "Left"/"Right"
                feature_vector = Normalize_Landmarks(hand_landmarks)
                label, confidence = Predict_Gesture(model, classes, feature_vector)

                # Tính vị trí hiển thị chữ neo vào cổ tay
                # Chiều cao h + Chiều rộng w khung camera thực tế
                h, w, _ = frame.shape
                wrist_px = (
                    int(hand_landmarks.landmark[0].x * w),
                    int(hand_landmarks.landmark[0].y * h) - 20,
                )

                # Cảnh báo lệch: MediaPipe thấy 1 tay nhưng model dự đoán nhãn
                # của tay kia (VD MediaPipe="Left" nhưng model="right_xxx")
                mismatch = mp_hand_label.lower() not in label.lower()

                if confidence >= CONFIDENCE_THRESHOLD:
                    text = f"{label} ({confidence*100:.1f}%) [{mp_hand_label}]"
                    color = (0, 0, 255) if mismatch else (0, 255, 0)
                else:
                    text = f"? ({label}, {confidence*100:.1f}%) [{mp_hand_label}]"
                    color = (0, 165, 255)  # cam - duoi nguong tin cay

                cv2.putText(
                    frame, text, wrist_px,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # FPS - de theo doi do tre pipeline that
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Phase 4 - Gesture Eval", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
