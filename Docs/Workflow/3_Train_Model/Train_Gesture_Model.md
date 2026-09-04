# Huấn luyện Model Gesture (MLP)

> Chi tiết kiến trúc, lý do lựa chọn, và quy trình huấn luyện model phân loại cử chỉ tay từ landmark. Thuộc Phase 3 của dự án **GestureVoice Desktop Controller**.

## 1. Input

| File | Shape | Mô tả |
|---|---|---|
| `X_gesture.npy` | `(N, 63)` | 21 landmark × (x, y, z), đã chuẩn hóa wrist-relative + scale từ Phase 1 |
| `Y_gesture.npy` | `(N,)` | Nhãn đã encode dạng số nguyên (0–9) |
| `Z_classes.npy` | `(10,)` | Bảng ánh xạ số → tên nhãn gốc, dùng để decode dự đoán |

## 2. Vì sao chọn MLP (không phải CNN/RNN/GCN...)

* Input đã là **feature vector tĩnh** do MediaPipe + bước chuẩn hóa ở Phase 1 trích sẵn, không mang cấu trúc không gian 2D (như ảnh) hay tính tuần tự thời gian (như audio) — nên các kiến trúc khai thác 2 đặc tính đó (CNN 2D, RNN, Transformer) không có gì thêm để học so với MLP trên cùng input này.
* Baseline đơn giản (KNN, MLP `(64,32)` của sklearn ở `Check_Dataset_Image.py`) đã đạt ≥85% accuracy — cho thấy giới hạn hiệu năng nằm ở lượng thông tin sẵn có trong 63 số, không nằm ở khả năng biểu diễn của mạng. Tăng độ phức tạp kiến trúc trong trường hợp này tăng rủi ro overfit nhiều hơn là tăng accuracy.
* Quy mô dữ liệu (~3000 mẫu, ~300 mẫu/lớp) phù hợp với một mạng nhỏ, không đủ lớn để nuôi các kiến trúc nhiều tham số hơn (GCN, attention) học tốt hơn MLP.

## 3. Kiến trúc (chốt)

`Input(63) → Dense(128) + BatchNorm + ReLU + Dropout(0.3) → Dense(64) + BatchNorm + ReLU + Dropout(0.3) → Dense(10) + Softmax`

* **2 hidden layer (128 → 64), không thêm tầng thứ 3**: input 63 chiều tĩnh + ~3000 mẫu là quy mô nhỏ, baseline 2 lớp đã ≥85% accuracy — thêm tầng chỉ tăng tham số mà không có cơ sở để tin nó học thêm được gì, rủi ro overfit tăng nhiều hơn lợi ích. Chỉ cân nhắc thêm tầng nếu sau khi train thấy **underfit thật sự** (train accuracy cũng thấp, không riêng val), và phải so sánh bằng thực nghiệm chứ không mặc định thêm là tốt hơn.
* **BatchNorm**: giúp hội tụ nhanh, ổn định hơn giữa các batch nhỏ — trước đây thường bị né khi deploy MCU vì tốn phép tính runtime, nhưng ở đây model chạy trên laptop nên dùng thoải mái.
* **Dropout(0.3)** mỗi lớp: chống overfit, quan trọng vì số mẫu/lớp không lớn.

## 4. Cấu hình huấn luyện (chốt)

* **Optimizer**: Adam, loss `sparse_categorical_crossentropy` (nhãn dạng số nguyên, không cần one-hot).
* **Chia tập**: 80/10/10 (train/val/test), `stratify` theo nhãn để giữ tỉ lệ lớp đồng đều ở cả 3 tập.
* **Callbacks sử dụng**:
  * `EarlyStopping` (theo dõi `val_loss`) — dừng sớm khi val_loss không cải thiện, tránh train dư epoch không cần thiết trên model nhỏ.
  * `ModelCheckpoint` — ghi checkpoint tốt nhất ra đĩa theo từng epoch, không mất kết quả nếu quá trình train bị gián đoạn.
  * `CSVLogger` — ghi log loss/accuracy theo epoch ra file, dùng để vẽ learning curve đưa vào báo cáo.
  * `TerminateOnNaN` — dừng ngay nếu loss "nổ" thành NaN, tránh train uổng công.
  * **Không dùng**: `ReduceLROnPlateau`, `LearningRateScheduler`, `TensorBoard`, `BackupAndRestore` — model nhỏ, train nhanh (vài chục giây tới vài phút), các callback này không mang lại lợi ích tương xứng với độ phức tạp thêm vào.
* **Class weight**: không bật mặc định — dataset đã được kiểm soát cân bằng từ khâu thu thập (`check_counts()` ở Phase 1). Chỉ bật `class_weight="balanced"` nếu kiểm tra số lượng mẫu thực tế theo lớp (`np.bincount(y)`) cho thấy lệch rõ rệt.

## 5. Đọc kết quả đánh giá

* **Accuracy tổng thấp nhưng đồng đều giữa các lớp**: dấu hiệu underfit thật sự — đây là trường hợp duy nhất đáng cân nhắc thêm tầng thứ 3 (VD 128→64→32→10), thử nghiệm và so sánh val accuracy trước khi kết luận, hoặc do landmark của 2 gesture nào đó vốn quá giống nhau về hình học.
* **1–2 lớp cụ thể có recall/precision thấp rõ rệt trong khi các lớp còn lại tốt**: nhìn vào confusion matrix để biết lớp đó bị nhầm sang lớp nào — nếu là 2 gesture hình dạng gần giống (VD `right_four` vs `right_five`), cân nhắc thu thêm mẫu đa dạng góc/khoảng cách cho riêng 2 lớp đó thay vì đổi kiến trúc.
* **Train accuracy cao, val/test accuracy thấp hơn rõ rệt**: dấu hiệu overfit — tăng Dropout hoặc giảm số unit mỗi lớp trước khi nghĩ đến thu thêm dữ liệu.

## 6. Deliverables

* `gesture_mlp_model.keras`: model đã train, sẵn sàng nạp lại ở Phase 4 (`eval_gesture_webcam.py`).
* Báo cáo `classification_report` + confusion matrix trên tập test.
* `Z_classes.npy` (đã có từ Phase 2) tiếp tục được dùng để decode `y_pred` (số nguyên) ngược về tên gesture khi chạy real-time.
