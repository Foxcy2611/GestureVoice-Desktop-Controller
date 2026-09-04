"""
====================
Tên script: Collect_Gesture_Data.py
Tác dụng: Thu landmark hai bàn tay từ webcam và lưu dữ liệu gesture có nhãn vào CSV.
====================
"""

"""
Phase 1 - GestureVoice Desktop Controller
Script thu thập dữ liệu landmark bàn tay bằng webcam + MediaPipe Hands.

Cách dùng:
    python collect_data.py

Trong cửa sổ webcam:
    - Giơ tay trước camera, giữ ổn định đúng cử chỉ muốn thu.
    - Bấm phím tương ứng để lưu 1 mẫu vào CSV (xem bảng phím bên dưới).
    - Bấm 'q' để thoát.

Bảng phím gán nhãn:
    0 -> fist   (nắm tay / 0 ngón)
    1 -> one
    2 -> two
    3 -> three
    4 -> four
    5 -> five
    l -> like
    o -> ok

File dữ liệu sẽ được lưu tại: Gesture_Data.csv (cùng thư mục với script)
Mỗi dòng = 63 giá trị tọa độ landmark (đã chuẩn hóa) + 1 nhãn.
"""

import os
import cv2
import csv
import time
import numpy as np
import mediapipe as mp

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "Landmarks_Normalize", "Gesture_Data_656.csv")

# Dict ánh xạ nút bấm
# ord trả về ký tự ascii của nút bấm
# Vì CSV trả về ASCII
Key_To_Label = { 
    ord("0"): "fist",
    ord("1"): "one",
    ord("2"): "two",
    ord("3"): "three",
    ord("4"): "four",
    ord("5"): "five",
    ord("l"): "like",
    ord("o"): "ok",
}

# Case nào được phép thu cho từng tay (Thiết kế 10 case: 2 trái + 8 phải)
Allowed_Gestures_Per_Hand = {
    "Left": {"one", "two"},
    "Right": {"fist", "one", "two", "three", "four", "five", "like", "ok"},
}

# Số điểm Lanmark trên 1 bàn tay
Num_Landmark = 21

# ======= Hàm chuẩn hóa landmark =======
# Nhận input thô rồi xuất ra 63 số đã chuẩn hóa độc lập với vị trí/khoảng cách tay
# Chuyển 21 landmark thành mảng np shape (21, 3)
# Preprocess

def Normalize_Landmarks(hand_landmarks):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # 1. Lấy điểm gốc là cổ tay chỉ số 0
    writs = coords[0]
    coords -= writs

    # 2. Chuẩn hóa kích thước bàn tay về 1 tỉ lệ
    # Để khi giơ tay gần hay xa khiến AI nhầm lẫn
    # Điểm xa nhất luôn cách cổ tay 1 đơn vị [-1 ; 0 ; 1]
    
    # Tính chuẩn hóa từng hàng rồi tìm điểm max (axis=1 là hàng)
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist >= 1e-6:
        coords /= max_dist

    return coords.flatten()


# ======= Khởi tạo MediaPipe Hands =======
mp_hands   = mp.solutions.hands
mp_drawing_ultis = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ======= Khởi tạo CSV thu dataset =======
# Tạo trước file CSV
def Init_CSV():
    if not os.path.exists(CSV_PATH):
        header = ["label"] # Cột đầu tiên là "label"

        for idx in range(Num_Landmark):
            header += [f"x{idx}", f"y{idx}", f"z{idx}"]

        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        print(f"[INFO] Khởi tạo File CSV thành công tại {CSV_PATH} !!")
    else:
        print(f"[INFO] Khởi tạo File CSV thất bại")

# Hàm thêm dữ liệu
def Append_Sample(label, landmark_vct):
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label] + landmark_vct.tolist())

# Đếm số lượng mẫu, in thống kê
def Count_Samples_per_label():
    # counts[label] = int
    counts = {}
    if not os.path.exists(CSV_PATH):
        return counts

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        data = csv.reader(f)
        # Bỏ qua header
        next(data, None)

        for row in data:
            if not row: # Rỗng thì bỏ qua
                continue
            else:
                label = row[0] # Lấy label
                # Nếu chưa có mà đã xuất hiện thì cho = 1
                counts[label] = counts.get(label, 0) + 1 

    return counts

def main():
    Init_CSV()

    # Đếm số mẫu đã có sẵn theo từng nhãn (nếu CSV đã có data từ trước),
    # để log hiển thị đúng số thứ tự tiếp theo chứ không đếm lại từ 0
    label_counts = Count_Samples_per_label()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Lỗi! Không mở được webcam. Kiểm tra kết nối Camera")
        return

    print("=" * 60)
    print("PHASE 1A - THU THẬP DATASET CỬ CHỈ TAY")
    print("=" * 60)
    print("Phím tương ứng với nhãn")
    for k, v in Key_To_Label.items():
        print(f"   [{chr(k)}]  ->  {v}")
    print("   [q]  ->  Thoat chuong trinh")
    print("=" * 60)

    last_saved_label = None
    last_saved_time = 0
    flash_until = 0  # thoi diem het hieu ung flash bao da luu mau

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Không đọc được frame ảnh từ webcam")
            break

        # Lật ảnh dễ nhìn như soi gương
        frame = cv2.flip(frame, 1)

        # MediaPipe cần sử dụng bảng màu RGB (OpenCV default dùng BGR)
        rgb_fame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_fame)

        # List gồm: handedness_label (nhãn bàn tay) + norm vector landmarks
        # Vdu:     ("Left",  array([x0,y0,z0, x1,y1,z1, ..., x20,y20,z20])),
        detected_hands_data = []

        if results.multi_hand_landmarks:
            # Duyệt qua 21 tọa độ điểm ; Nhãn left/right
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):

                # Vẽ landmark lên khung hình kiểm tra tính trực quan
                mp_drawing_ultis.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    # Trả về kiểu vẽ mặc định các điểm mốc bàn tay
                    mp_drawing_styles.get_default_hand_landmarks_style(), 
                    # Trả về kiểu vẽ mặc định các điểm nối bàn tay
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                hand_label  = handedness.classification[0].label # Left / Right
                norm_vector = Normalize_Landmarks(hand_landmarks)
                detected_hands_data.append((hand_label, norm_vector)) # Lưu vào list

        # ---- Hiển thị thông tin trên khung hình ----
        y_offset = 30
        cv2.putText(
            frame,
            f"Số tay đang phát hiện: {len(detected_hands_data)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
 
        for idx, (hand_label, _) in enumerate(detected_hands_data):
            cv2.putText(
                frame,
                f"Tay {idx + 1}: {hand_label}",
                (10, y_offset + 30 * (idx + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
 
        # Hiệu ứng flash báo vừa lưu mẫu thành công
        if time.time() < flash_until:
            cv2.putText(
                frame,
                f"DA LUU: {last_saved_label}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in Key_To_Label:
            gesture = Key_To_Label[key]

            if not detected_hands_data:
                print("[WAR] Không phát hiện tay nào để lưu mẫu")
            else:
                # Check glitch: 2 tay nhưng trùng nhãn (VD Left+Left) là điều
                # vô lý về mặt sinh học -> MediaPipe đang bị nhiễu do góc
                # nghiêng quá gắt / che khuất ngón tay, không lưu mẫu lúc này.
                hand_labels_now = [h for h, _ in detected_hands_data]
                if len(hand_labels_now) != len(set(hand_labels_now)):
                    print(f"[WAR] Phat hien nhieu tay trung nhan ({hand_labels_now}) "
                          f"-> nghi ngo MediaPipe dang loi do goc nghieng qua gat, "
                          f"BO QUA khong luu. Chinh lai goc tay roi thu lai.")
                    continue
                
                saved_labels_this_frame = []
                for hand_label, vct in detected_hands_data:
                    allowed = Allowed_Gestures_Per_Hand.get(hand_label, set())
                    if gesture not in allowed:
                        print(f"[WAR] Tay '{hand_label}' không có case '{gesture}' "
                              f"(Chỉ cho phép: {sorted(allowed)}) -> BỎ QUA, không lưu")
                        continue
 
                    label = f"{hand_label.lower()}_{gesture}"
                    Append_Sample(label, vct)

                    label_counts[label] = label_counts.get(label, 0) + 1
                    last_saved_label = label
                    saved_labels_this_frame.append(label)
 
                if saved_labels_this_frame:
                    last_saved_time = time.time()
                    flash_until = last_saved_time + 0.4
                    # Log ngắn gọn, dễ đọc lướt: "<nhãn> -> #<số thứ tự hiện tại>"
                    log_parts = [f"{lb} -> #{label_counts[lb]}" for lb in saved_labels_this_frame]
                    print(f"[SAVE] {' | '.join(log_parts)}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # In thống kê
    print("\n" + "=" * 60)
    print("THỐNG KÊ SỐ MẪU ĐÃ THU THEO NHÃN")
    print("=" * 60)
    counts = Count_Samples_per_label()
    if not counts:
        print("Chưa có dữ liệu nào được lưu.")
    else:
        total = 0
        for label in sorted(counts.keys()):
            print(f"   {label:10s}: {counts[label]} mẫu")
            total += counts[label]
        print(f"   {'TỔNG':10s}: {total} mẫu")
    print(f"\nFile dữ liệu: {CSV_PATH}")

if __name__ == "__main__":
    main()

"""
============================================================
THỐNG KÊ SỐ MẪU ĐÃ THU THEO NHÃN
============================================================
   left_one  : 300 mẫu
   left_two  : 300 mẫu
   right_fist: 300 mẫu
   right_five: 300 mẫu
   right_four: 300 mẫu
   right_like: 300 mẫu
   right_ok  : 300 mẫu
   right_one : 300 mẫu
   right_three: 300 mẫu
   right_two : 300 mẫu
   TỔNG      : 3000 mẫu
"""
