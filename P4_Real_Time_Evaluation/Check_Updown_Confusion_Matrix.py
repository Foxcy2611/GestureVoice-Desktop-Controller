"""
====================
Tên script: Check_Updown_Confusion_Matrix.py
Tác dụng: Kiểm tra confusion matrix KWS tĩnh, tập trung vào nhầm lẫn UP và DOWN.
====================
"""

"""
Phase 4 - Buoc 2: Chan doan xem up/down nham nhau la do MODEL/DATA hay do
PIPELINE REAL-TIME.

Khong can noi gi ca - chi load lai dung TAP TEST TINH da dung luc train
(cung random_state=42, cung stratify) roi soi confusion matrix.

Neu tap test tinh nay da nham up/down nhieu -> dung la van de model/data
that, di tiep Buoc 3 (thu them background/noise-mixing/class_weight).

Neu tap test tinh nay gan nhu hoan hao cho up/down -> van de chi nam o
pipeline real-time (Eval_Voice_Mic.py), chua can dung den data/train lai.

Cach dung:
    python Check_UpDown_ConfusionMatrix.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import models

# ========= DUONG DAN - SUA LAI CHO KHOP MAY BAN =========
X_PATH = "2_Data_Preprocessing/Output_npy/X_voice.npy"
Y_PATH = "2_Data_Preprocessing/Output_npy/Y_labels.npy"
MODEL_PATH = "3_Train_Model/Output_KWS/KWS_DS_CNN_Model.keras"

CLASS_NAME = ["on", "off", "up", "down", "background"]
RANDOM_STATE = 42  # PHAI khop dung voi Train_Model_KWS.py de lay lai dung tap test cu

def Load_Data():
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    X = X[..., np.newaxis]  # them truc de khop dung input model (giong Load_Data() luc train)
    return X, Y


def lay_lai_tap_test(X, Y):
    """Lam DUNG HET giong Spilit_Data() trong Train_Model_KWS.py de dam bao
    lay lai chinh xac tap test da dung luc danh gia Phase 3."""
    X_train_np, X_temp, Y_train_np, Y_temp = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=Y
    )
    X_valid_np, X_test_np, Y_valid_np, Y_test_np = train_test_split(
        X_temp, 
        Y_temp, 
        test_size=0.5, 
        random_state=RANDOM_STATE,
        stratify=Y_temp
    )
    
    return X_test_np, Y_test_np


def main():
    print("Đang tải dữ liệu và mô hình...")
    X, Y = Load_Data()
    X_test, Y_test = lay_lai_tap_test(X, Y)
    print(f"  + Tập test: {X_test.shape[0]} mẫu (phải khớp đúng với Phase 3)")

    model = models.load_model(MODEL_PATH)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    print("\n" + "=" * 70)
    print("BÁO CÁO PHÂN LOẠI ĐẦY ĐỦ")
    print("=" * 70)
    print(classification_report(Y_test, y_pred, target_names=CLASS_NAME, digits=4))

    cm = confusion_matrix(Y_test, y_pred)
    print("=" * 70)
    print("MA TRẬN NHẦM LẪN (hàng = thực tế, cột = dự đoán)")
    print("=" * 70)
    header = "".join(f"{name:>12s}" for name in CLASS_NAME)
    print(" " * 12 + header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12d}" for v in row)
        print(f"{CLASS_NAME[i]:>12s}{row_str}")

    # ---- Tập trung riêng cặp up/down ----
    idx_up = CLASS_NAME.index("up")
    idx_down = CLASS_NAME.index("down")

    up_total = cm[idx_up].sum()
    up_to_down = cm[idx_up][idx_down]
    up_correct = cm[idx_up][idx_up]

    down_total = cm[idx_down].sum()
    down_to_up = cm[idx_down][idx_up]
    down_correct = cm[idx_down][idx_down]

    print("\n" + "=" * 70)
    print("CHẨN ĐOÁN RIÊNG CẶP UP/DOWN")
    print("=" * 70)
    print(f"  'up' thực tế   : đúng {up_correct}/{up_total} "
          f"({up_correct/up_total*100:.1f}%), bị nhầm thành 'down' {up_to_down} lần "
          f"({up_to_down/up_total*100:.1f}%)")
    print(f"  'down' thực tế : đúng {down_correct}/{down_total} "
          f"({down_correct/down_total*100:.1f}%), bị nhầm thành 'up' {down_to_up} lần "
          f"({down_to_up/down_total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("KẾT LUẬN")
    print("=" * 70)
    nham_lan_ratio = (up_to_down + down_to_up) / (up_total + down_total)
    if nham_lan_ratio < 0.05:
        print("  Tập test TĨNH gần như hoàn hảo cho up/down (nhầm lẫn < 5%).")
        print("  -> Vấn đề bạn gặp lúc dùng mic laptop NHIỀU KHẢ NĂNG nằm ở")
        print("     PIPELINE REAL-TIME (cắt cửa sổ, lệch phân bố), KHÔNG")
        print("     phải do mô hình/dữ liệu yếu. Ưu tiên tinh chỉnh Eval_Voice_Mic.py")
        print("     (energy gate, voting, RAW_CAPTURE_SEC) trước, CHƯA cần")
        print("     thu thêm dữ liệu hay train lại.")
    else:
        print(f"  Tập test TĨNH cũng đã nhầm lẫn up/down đáng kể ({nham_lan_ratio*100:.1f}%).")
        print("  -> Xác nhận đây là vấn đề MÔ HÌNH/DỮ LIỆU thật. Nên đi tiếp Bước 3:")
        print("     thu background đặc thù, code thêm noise-mixing lúc train,")
        print("     hoặc thử class_weight cho 2 lớp này trước khi thu thêm mẫu.")


if __name__ == "__main__":
    main()
