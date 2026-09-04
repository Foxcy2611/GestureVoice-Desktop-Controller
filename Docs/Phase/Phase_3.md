# Phase 3: Huấn luyện Model (Train Model)

> Thuộc dự án **GestureVoice Desktop Controller** - Huấn luyện 2 model độc lập từ dữ liệu `.npy` sinh ra ở Phase 2: MLP nhận diện cử chỉ tay và DS-CNN phân loại lệnh thoại.

## 1. Mục tiêu giai đoạn

Biến 5 file `.npy` (đặc trưng số, chưa có "trí tuệ" gì) thành 2 model đã học được ranh giới phân loại, đạt độ chính xác đủ tin cậy để dùng ở Phase 4 (đánh giá real-time qua webcam + micro).

* **Model 1 — Gesture**: phân loại 10 lớp (2 tay trái + 8 tay phải) từ 63 giá trị landmark.
* **Model 2 — Voice**: phân loại 5 lớp (`On`, `Off`, `Up`, `Down`, `Background`) từ chuỗi hệ số MFCC+delta+delta-delta.

Vì cả 2 model chạy trực tiếp trên laptop, ràng buộc chọn kiến trúc là **độ trễ đủ nhanh cho pipeline real-time + phù hợp quy mô dữ liệu hiện có**, không phải ràng buộc bộ nhớ/flash của vi điều khiển.

## 2. Input / Output của giai đoạn

| | Input (từ Phase 2) | Output |
|---|---|---|
| Gesture | `X_gesture.npy`, `Y_gesture.npy`, `Z_classes.npy` | `gesture_mlp_model.h5` (hoặc `.keras`) |
| Voice | `X_voice.npy`, `Y_labels.npy` | `KWS_DS_CNN_Model.keras` |

Chi tiết kiến trúc, lý do lựa chọn, và quy trình huấn luyện của từng model được tách riêng thành 2 tài liệu:

* [`Train_Gesture_Model.md`](./Train_Gesture_Model.md) — MLP cho gesture.
* [`Train_Voice_Model.md`](../Workflow/3_Train_Model/Train_Voice_Model.md) — DS-CNN cho KWS voice.

## 3. Quy trình chung áp dụng cho cả 2 model

1. **Load `.npy`** từ `Output_npy/` (không đọc lại CSV/wav).
2. **Chia tập train/val/test** (thực hiện ở bước này, không phải Phase 2) — dùng `stratify` theo nhãn để không lệch phân bố lớp.
3. **Huấn luyện** với callback `EarlyStopping` theo dõi `val_loss`, tránh train quá số epoch cần thiết.
4. **Đánh giá** trên tập test: accuracy tổng, `classification_report` theo từng lớp, confusion matrix — để phát hiện lớp nào bị nhầm với lớp nào (VD 2 gesture hình dạng gần giống nhau).
5. **Lưu model** + file phụ trợ cần cho việc giải mã dự đoán (`Z_classes.npy` cho gesture; bảng ánh xạ cố định cho voice không cần lưu thêm vì đã fix sẵn).

## 4. Kết quả cuối cùng (Deliverables)

Sau khi Phase 3 hoàn tất, dự án sẽ có:

* 2 file model đã huấn luyện, đạt ngưỡng accuracy chấp nhận được trên tập test (mục tiêu tham chiếu: ≥85%, dựa trên baseline sklearn đã đo ở Phase 1).
* Báo cáo đánh giá (classification report + confusion matrix) cho từng model, làm căn cứ quyết định có cần thu thêm dữ liệu hay đổi kiến trúc trước khi sang Phase 4 hay không.
