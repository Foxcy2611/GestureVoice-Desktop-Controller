# Phase 1 — Tiền xử lý `Gesture_Data.csv` → `.npy` (Input cho MLP Gesture)

> Tài liệu mô tả pipeline chuyển đổi dữ liệu landmark tay đã thu thập (`Gesture_Data.csv`) thành tensor `.npy` sẵn sàng cho huấn luyện MLP ở Phase 2. Kỹ thuật chuẩn hóa sử dụng trong `Collect_Image_Data.py` được đối chiếu với các nghiên cứu đã công bố, không tự phát minh.

---

## 1. Bối cảnh dữ liệu đầu vào

`Gesture_Data.csv` được sinh ra bởi `Collect_Image_Data.py`, mỗi dòng gồm:

```
label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
```

- **21 landmark/tay × 3 tọa độ (x, y, z) = 63 giá trị đặc trưng.**
- `label` là chuỗi ghép `{tay}_{cử_chỉ}` (VD: `left_one`, `right_fist`) — 10 lớp tổng cộng (2 lớp tay trái + 8 lớp tay phải), khớp thiết kế nêu trong `README.md` mục 3.
- Landmark đã được chuẩn hóa **tại thời điểm thu thập** (hàm `Normalize_Landmarks`), theo 2 bước:
  1. **Translational normalization**: trừ toạ độ mọi landmark cho toạ độ landmark cổ tay (index 0) → gốc toạ độ luôn là cổ tay.
  2. **Scale normalization**: chia toàn bộ toạ độ cho khoảng cách Euclid lớn nhất tính từ cổ tay đến landmark xa nhất → điểm xa nhất luôn cách gốc đúng 1 đơn vị.

## 2. Căn cứ khoa học cho phương pháp chuẩn hóa

Đây **không phải kỹ thuật tự đặt ra**, mà là pattern chuẩn trong các công trình dùng MediaPipe Hands cho gesture/sign-language recognition:

1. Gil-Martín, M., Marini, M., Martín-Fernández, I., Esteban-Romero, S., & Cinque, L. (2025). *Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks.* Proceedings of ICAART 2025, Vol. 3, pp. 24–30, SciTePress.
   - So sánh nhiều biến thể chuẩn hóa toạ độ landmark; biến thể lấy **landmark cổ tay làm điểm tham chiếu (per-frame wrist reference)** đạt độ chính xác ~83.7%, cho thấy đây là baseline được cộng đồng nghiên cứu công nhận, không phải cách làm tùy tiện.

2. *Enhancing ASL Recognition with GCNs and Successive Residual Connections* (arXiv:2408.09567).
   - Mô tả thuật toán tiền xử lý gồm đúng 2 bước mà `Normalize_Landmarks` đang dùng: (a) trừ theo landmark gốc để chuẩn hoá vị trí (translational normalization), (b) chia theo khoảng cách lớn nhất để chuẩn hoá tỉ lệ (scale normalization) — cùng công thức `d_max = max(‖x_i'‖)` và scale theo `d_max`.

3. *Dynamic Hand Gesture Recognition Using MediaPipe and Transformer* (MDPI, 2025).
   - Xác nhận: mỗi tay MediaPipe trả về 21 landmark, toạ độ x,y nằm trong [0,1], z là độ sâu tương đối so với landmark 0 (cổ tay) — đúng với cấu trúc 63 cột trong `Gesture_Data.csv`.

**Kết luận cho báo cáo:** việc chuẩn hóa wrist-relative + scale-by-max-distance là kỹ thuật đã được kiểm chứng trong tài liệu học thuật cho bài toán gesture recognition dựa trên MediaPipe landmark, giúp mô hình bất biến với khoảng cách tay–camera và vị trí tay trong khung hình.

## 3. Vì sao bước tiền xử lý ở đây KHÔNG normalize lại lần 2

Vì `Collect_Image_Data.py` đã chuẩn hóa ngay lúc ghi CSV (thư mục `Landmarks_Normalize/` xác nhận điều này), bước tiền xử lý trước khi train chỉ cần:

- Audit lại dữ liệu (đã làm ở `Check_Dataset_Image.py`: đếm mẫu, check NaN, check trùng lặp, baseline KNN/MLP).
- Encode label, split, xuất `.npy`.

Việc chuẩn hóa lại từ đầu (VD min-max theo cột) sẽ phá vỡ tính chất hình học đã được thiết lập (tỉ lệ giữa các landmark trong cùng 1 bàn tay), nên **không thực hiện lại**.

## 4. Pipeline xuất `.npy`

Bước này **không đơn thuần là "copy cột CSV sang npy"**. `.npy` không phải định dạng khác của cùng dữ liệu — nó thực hiện 3 việc mà CSV không tự có:

- **Validate cấu trúc**: đảm bảo đúng 64 cột (1 label + 63 feature) và đủ 10 nhãn trước khi cho qua bước sau — CSV không tự kiểm tra được điều này, phải code kiểm tra.
- **Encode label**: `y` trong CSV đang là chuỗi (`"left_one"`, `"right_fist"`...) — mạng neuron không nhận string, phải map sang số nguyên (`LabelEncoder`) và **lưu lại bảng ánh xạ số ↔ tên** (`label_classes.npy`), vì thứ tự encode phụ thuộc dữ liệu lúc chạy, không cố định — nếu không lưu, lúc deploy real-time sẽ không biết `y_pred = 3` là gesture gì.
- **Đổi kiểu dữ liệu & định dạng lưu trữ**: từ `pandas.DataFrame` (text-based, đọc/ghi chậm) sang `numpy.ndarray` dạng nhị phân (`float32` cho X, `int` cho y) — nhẹ hơn, load nhanh hơn nhiều lần khi train lặp lại nhiều epoch, và là định dạng chuẩn mà TensorFlow/Keras, scikit-learn nhận trực tiếp làm input.

```
                           Gesture_Data.csv
                                 │
                                 ▼
         Load + validate: đủ 64 cột (label + 63 feature), đủ 10 nhãn
                                 |
                                 ▼
            Tách X (63 cột feature) / y (cột label, dạng chuỗi)
                                 |
                                 ▼
      LabelEncoder: y_str → y_int, lưu bảng ánh xạ (label_classes.npy)
                                 |
                                 ▼
               Ghi ra .npy: X.npy, y.npy, label_classes.npy 
```

> Quá trình xuất ra `.npy` của model hình ảnh xem tại đây [Pre Model Image](../../2_Data_Preprocessing/Preprocess_Gesture_CSV.py)

## 5. Tài liệu tham khảo

1. Gil-Martín, M. et al. (2025). *Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks.* ICAART 2025, Vol. 3, pp. 24–30. https://www.scitepress.org/PublishedPapers/2025/130535/
2. *Enhancing ASL Recognition with GCNs and Successive Residual Connections.* arXiv:2408.09567. https://arxiv.org/pdf/2408.09567
3. *Dynamic Hand Gesture Recognition Using MediaPipe and Transformer.* MDPI Engineering Proceedings, 2025. https://www.mdpi.com/2673-4591/108/1/22