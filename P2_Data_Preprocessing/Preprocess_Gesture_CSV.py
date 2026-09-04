"""
====================
Tên script: Preprocess_Gesture_CSV.py
Tác dụng: Kiểm tra CSV gesture, mã hóa nhãn và xuất tensor X/Y cùng bảng lớp Z.
====================
"""

"""
Đơn thuần dễ hiểu, từ file csv thu thập data, nó xuất ra các file .npy sau đây
+ X_gesture: Ma trận đầu vào, là 1 ma trận 3000 x 63 (3000 mẫu x 63 hệ số tọa độ x, y, z)
=> Khi này, input đầu vào mạng neuron dạng 1x63
+ Y_gesture: Vector đáp án, vs mỗi hàng sẽ là ứng với label của nó (left_one, ... ???)
+ Z_classes: Tổng hợp số nhãn
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(
    BASE_DIR,
    "P1_Data_Collection",
    "Dataset",
    "Landmarks_Normalize",
    "Gesture_Data.csv",
)
OUT_DIR = os.path.join(BASE_DIR, "P2_Data_Preprocessing", "Output_npy")

NUM_LANDMARK = 21
NUM_FEATURES = NUM_LANDMARK * 3  # 63 hệ số ứng vs x, y, z
EXPECTED_LABELS = 10 # Tổng 10 trường hợp

# Load CSV và kiểm tra lỗi
def Load_And_Validate(csv_path):
    df = pd.read_csv(csv_path)

    expected_cols = 1 + NUM_FEATURES  # label + 63 feature
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"[LỖI] Số cột không khớp: Có {df.shape[1]}, kỳ vọng {expected_cols}"
        )

    n_labels = df["label"].nunique()
    if n_labels != EXPECTED_LABELS:
        print(f"[CẢNH BÁO] Số nhãn hiện có là {n_labels}, kỳ vọng {EXPECTED_LABELS}")

    if df.isna().sum().sum() > 0:
        raise ValueError("[LỖI] Còn giá trị NaN, hãy chạy lại Check_Dataset_Image.py trước")

    return df

# Tách 63 cột tọa độ ra khỏi label
# Mã hóa label y: Vì máy tính ko hiểu chuỗi string như "left_one"
def Encode_Full(df):
    # Lấy toàn bộ cột trừ cột "label"
    feature_cols = [c for c in df.columns if c != "label"]
    # Chuyển định dạng float64 mặc định của Py xuống float32
    X = df[feature_cols].values.astype(np.float32)
    # Bốc riêng cột label
    y_str = df["label"].values

    # Sử dụng hàm labelencoder, chuyên chuyển chữ thành số
    le = LabelEncoder()
    # Tìm các nhãn, xếp thành theo thứ tự và thay chúng = số
    y = le.fit_transform(y_str).astype(np.int64)

    return X, y, le.classes_


def Save_npy(out_dir, data_dict):
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in data_dict.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"   Đã lưu {name}.npy  shape={np.asarray(arr).shape}")


def main():
    print("=" * 60)
    print("TIỀN XỬ LÝ Gesture_Data.csv -> .npy (tổng thể)")
    print("=" * 60)

    df = Load_And_Validate(CSV_PATH)
    X, y, label_classes = Encode_Full(df)

    print(f"\nX: {X.shape}   y: {y.shape}")
    print(f"Các nhãn (thứ tự encode): {list(label_classes)}")

    Save_npy(OUT_DIR, {
        "X_gesture": X,
        "Y_gesture": y,
        "Z_classes": label_classes,
    })

    print("\nHOÀN TẤT. Việc chia train/val/test sẽ thực hiện ở Phase 2.")


if __name__ == "__main__":
    main()

"""
============================================================
TIỀN XỬ LÝ Gesture_Data.csv -> .npy (tổng thể)
============================================================

X: (3000, 63)   y: (3000,)
Các nhãn (thứ tự encode): ['left_one', 'left_two', 'right_fist', 'right_five', 'right_four', 'right_like', 'right_ok', 'right_one', 'right_three', 'right_two']
   Đã lưu X_gesture.npy  shape=(3000, 63)
   Đã lưu Y_gesture.npy  shape=(3000,)
   Đã lưu Z_classes.npy  shape=(10,)

HOÀN TẤT. Việc chia train/val/test sẽ thực hiện ở Phase 2.
"""
