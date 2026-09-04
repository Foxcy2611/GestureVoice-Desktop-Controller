# Phase 2: Tiền xử lý dữ liệu (Data Preprocessing)

> Thuộc dự án **GestureVoice Desktop Controller** - Chuyển dữ liệu thô đã thu ở Phase 1 (`Gesture_Data.csv` + `Voice_Data_Raw/`) thành tensor `.npy` sẵn sàng cho huấn luyện ở Phase 3.

## 1. Mục tiêu giai đoạn

Phase 1 chỉ tạo ra dữ liệu ở dạng "con người/hệ thống file đọc được" (CSV, thư mục `.wav`) — chưa phải định dạng mà model neural network nhận trực tiếp. Phase 2 thu hẹp khoảng cách đó cho cả 2 phân hệ:

* **Gesture**: landmark trong CSV **đã được chuẩn hóa từ lúc thu thập** (wrist-relative + scale, xem `Collect_Image_Data.py` ở Phase 1) — Phase 2 không chuẩn hóa lại, chỉ validate cấu trúc, encode nhãn dạng chuỗi sang số, và xuất ra `.npy`.
* **Voice**: file `.wav` thô 1 giây **chưa qua trích đặc trưng** — Phase 2 mới là nơi thực hiện trích MFCC + delta + delta-delta (39 hệ số) và gán nhãn cố định 5 lớp.

> Lưu ý: bước tiền xử lý CSV gesture trước đây có thảo luận lồng trong tài liệu Phase 1, nhưng về bản chất quy trình nó thuộc Phase 2 (đúng như cây thư mục `2_Data_Preprocessing/`).

## 2. Cấu trúc thư mục đầu ra

```
2_Data_Preprocessing/
├── Preprocess_Gesture_CSV.py     <- Đọc Gesture_Data.csv, validate + encode + export
├── Preprocess_KWS_Raw.py         <- Đọc Voice_Data_Raw/, trích MFCC+delta+delta-delta + augmentation laptop + export
└── Output_npy/
    ├── X_gesture.npy             <- Feature landmark gesture, shape (N_gesture, 63)
    ├── Y_gesture.npy             <- Nhãn gesture đã encode (số nguyên), shape (N_gesture,)
    ├── X_voice.npy               <- Feature MFCC+delta+delta-delta, shape (N_voice, T, 39)
    ├── Y_labels.npy              <- Nhãn voice (0=On,1=Off,2=Up,3=Down,4=Background)
    └── Z_classes.npy             <- Bảng ánh xạ số→tên nhãn gesture (dùng để decode lúc eval/deploy)
```

## 3. Nhánh xử lý Gesture (`Preprocess_Gesture_CSV.py`)

1. Load `Gesture_Data.csv`, validate đúng 64 cột (1 label + 63 feature) và đủ 10 nhãn.
2. Kiểm tra không còn `NaN` (đã audit ở `Check_Dataset_Image.py`, bước này validate lại lần cuối trước khi export).
3. Tách `X` (63 cột feature, ép kiểu `float32`) và `y` (cột `label`, dạng chuỗi).
4. `LabelEncoder` chuyển `y` chuỗi (`left_one`, `right_fist`...) sang số nguyên → lưu `Y_gesture.npy`; lưu bảng lớp gốc (`le.classes_`) → `Z_classes.npy` để giải mã ngược khi cần biết `y_pred = 3` là gesture gì.
5. Ghi `X_gesture.npy`, `Y_gesture.npy`, `Z_classes.npy` — **dữ liệu tổng thể, chưa chia train/val/test** (việc chia tập thực hiện ở Phase 3 lúc train, xem `Train_Gesture_Model.md`).

## 4. Nhánh xử lý Voice (`Preprocess_KWS_Raw.py`)

1. Duyệt qua `Voice_Data_Raw/{On,Off,Up,Down}/` và `Voice_Data_Raw/Background/{Kaggle,INMP441,Laptop}/` — tổng 15.000 file `.wav` dài 1 giây. Mỗi lớp gồm 3.000 file; trong đó có 30 file tự thu bằng micro laptop.
2. Với mỗi file: resample về tần số lấy mẫu cố định (nếu chưa đồng nhất), trích **MFCC** (thường 13 hệ số) + đạo hàm bậc 1 (**delta**) + đạo hàm bậc 2 (**delta-delta**) → tổng 39 hệ số/khung thời gian, đúng cấu hình đã nêu ở README mục 4.
3. Gán nhãn cố định theo thư mục nguồn (không cần `LabelEncoder` vì bảng ánh xạ đã chốt sẵn):

   | Thư mục nguồn | Nhãn số |
   |---|---|
   | `On/` | 0 |
   | `Off/` | 1 |
   | `Up/` | 2 |
   | `Down/` | 3 |
   | `Background/Kaggle/` + `Background/INMP441/` + `Background/Laptop/` | 4 |

4. Với mỗi `laptop_*.wav`, giữ 1 feature gốc và tạo thêm 7 feature augmentation tĩnh. Vì áp dụng cho cả 5 lớp, output có `15000 + (5 × 30 × 7) = 16050` feature, vẫn chỉ ghi hai file `X_voice.npy` và `Y_labels.npy` để `Train_Model_KWS.py` dùng nguyên vẹn.
5. Ghi `X_voice.npy` (shape `(16050, 51, 39)`) và `Y_labels.npy` (nhãn số tương ứng).

## 5. Kết quả cuối cùng (Deliverables)

Sau khi Phase 2 hoàn tất, dự án sẽ có đủ 5 file `.npy` trong `Output_npy/` — input trực tiếp cho Phase 3, không cần đọc lại CSV hay quét thư mục `.wav` nữa.
