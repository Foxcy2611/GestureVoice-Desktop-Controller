"""
====================
Tên script: Check_Dataset_Image.py
Tác dụng: Kiểm tra cấu trúc, độ cân bằng và chất lượng dataset gesture trước khi train.
====================
"""

"""
GestureVoice Desktop Controller - Kiem tra chat luong dataset Gesture_Data.csv
Chay sau khi thu du du lieu, TRUOC khi dau tu train model chinh thuc.

"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(
    BASE_DIR,
    "P1_Data_Collection",
    "Dataset",
    "Landmarks_Normalize",
    "Gesture_Data.csv",
)
EXPECTED_PER_LABEL = 300
DUP_DIST_THRESHOLD = 0.02  # nguong khoang cach Euclid coi la "gan nhu trung lap"


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


def check_counts(df):
    print("=" * 60)
    print("1. SỐ LƯỢNG MẪU THEO NHÃN")
    print("=" * 60)
    counts = df["label"].value_counts().sort_index()
    for label, cnt in counts.items():
        flag = "ĐỦ" if cnt >= EXPECTED_PER_LABEL * 0.95 else "THIẾU"
        print(f"   {label:15s}: {cnt:4d}  [{flag}]")
    print(f"   Tổng số nhãn: {len(counts)} (kỳ vọng 10)")
    print(f"   Tổng số mẫu : {len(df)} (kỳ vọng {EXPECTED_PER_LABEL * 10})")


def check_nan(df):
    print("\n" + "=" * 60)
    print("2. KIỂM TRA NaN / GIÁ TRỊ LỖI")
    print("=" * 60)
    n_nan = df.isna().sum().sum()
    if n_nan == 0:
        print("   Không có giá trị NaN. OK.")
    else:
        print(f"   [CẢNH BÁO] Phát hiện {n_nan} giá trị NaN -> cần làm sạch trước khi train")
        bad_rows = df[df.isna().any(axis=1)]
        print(f"   Số dòng bị ảnh hưởng: {len(bad_rows)}")


def check_duplicates(df):
    print("\n" + "=" * 60)
    print(f"3. KIỂM TRA GẦN TRÙNG LẶP (ngưỡng khoảng cách < {DUP_DIST_THRESHOLD})")
    print("=" * 60)
    feature_cols = [c for c in df.columns if c != "label"]

    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label][feature_cols].values
        n = len(sub)
        if n < 2:
            continue

        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            sub_sample = sub[idx]
        else:
            sub_sample = sub

        dup_count = 0
        total_pairs = 0
        for i in range(len(sub_sample)):
            for j in range(i + 1, len(sub_sample)):
                dist = np.linalg.norm(sub_sample[i] - sub_sample[j])
                total_pairs += 1
                if dist < DUP_DIST_THRESHOLD:
                    dup_count += 1

        pct = 100 * dup_count / total_pairs if total_pairs > 0 else 0
        flag = "OK" if pct < 5 else "CẢNH BÁO - quá nhiều mẫu giống nhau"
        print(f"   {label:15s}: {pct:5.1f}% cặp gần trùng lặp  [{flag}]")


def check_baseline_model(df):
    print("\n" + "=" * 60)
    print("4. KIỂM TRA THỬ - KNN + MLP (chưa phải model chính thức)")
    print("=" * 60)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)

    print(f"   KNN độ chính xác (test 20%): {acc_knn * 100:.2f}%")
    print(f"   MLP độ chính xác (test 20%): {acc_mlp * 100:.2f}%")

    if acc_knn < 0.85:
        print("   [CẢNH BÁO] KNN thấp -> nghi ngờ dữ liệu: nhãn bị lẫn,")
        print("              landmark nhiễu, hoặc 2 gesture quá giống nhau về hình học.")
    if acc_mlp < 0.85:
        print("   [CẢNH BÁO] MLP thấp -> có thể do dữ liệu, cũng có thể do MLP chưa hội tụ")
        print("              (thử tăng max_iter hoặc chỉnh hidden_layer_sizes trước khi kết luận).")
    if acc_knn >= 0.85 and acc_mlp >= 0.85:
        print("   OK - cả 2 baseline đều tốt, dữ liệu sẵn sàng để train model chính thức.")

    print("\n   Chi tiết MLP theo từng nhãn:")
    print(classification_report(y_test, y_pred_mlp, zero_division=0))

def main():
    df = load_data()
    check_counts(df)
    check_nan(df)
    check_duplicates(df)
    check_baseline_model(df)


if __name__ == "__main__":
    main()
