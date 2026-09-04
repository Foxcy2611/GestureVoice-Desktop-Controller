# Phase 1: Thu thập Dataset (Data Collection)

> Thuộc dự án **GestureVoice Desktop Controller** - Giai đoạn khởi đầu chuẩn bị dữ liệu hình ảnh và âm thanh.
> Chi tiết quy trình tổ chức thư mục và các bước tiền xử lý xem tại `README.md`.

## 1. Mục tiêu giai đoạn

Giai đoạn này không tạo ra mô hình AI mà tập trung chuẩn bị "nguyên liệu" sạch và đa dạng cho 2 phân hệ:

* **Hình ảnh**: bộ dữ liệu landmark tay có nhãn, lưu thành CSV, phục vụ huấn luyện model MLP nhận diện cử chỉ. Dùng landmark (63 số/mẫu) thay vì ảnh thô, phù hợp hướng TinyML/Edge AI.
* **Âm thanh**: bộ 15.000 mẫu `.wav` dài 1 giây cho 5 lớp KWS. Mỗi keyword có 2.970 mẫu Google Speech Commands + **30 mẫu tự thu bằng micro laptop**; lớp `Background` có 1.485 Kaggle + 1.485 INMP441 + **30 mẫu laptop**. Phần laptop là bắt buộc để giảm lệch miền microphone và tránh model đoán sai khi chạy thực tế.

Nguyên tắc xuyên suốt: **garbage in, garbage out** — dữ liệu thu cẩu thả sẽ giới hạn trần năng lực của model, bất kể kiến trúc mạng ở Phase 2 tốt đến đâu.

## 2. Phạm vi công việc

* Xây dựng script thu thập cử chỉ tay qua webcam (OpenCV + MediaPipe Hands), gán nhãn thủ công bằng phím bấm.
* Xây dựng script bốc mẫu + cắt ghép tự động cho dữ liệu âm thanh từ Kaggle/INMP441, đồng thời thu bổ sung 30 mẫu 1 giây/lớp bằng đúng micro laptop triển khai.
* Kiểm tra chất lượng dữ liệu sau thu thập (số lượng, cân bằng lớp, lỗi/nhiễu) trước khi bước sang tiền xử lý → `.npy`.

## 3. Kết quả đầu ra (Deliverables)

Sau khi Phase 1 hoàn tất, hệ thống sẽ sở hữu:

* Các script thực thi: `Collect_Gesture_Data.py`, `Collect_KWS_Data.py` (bốc/cắt nguồn) và `Record_KWS_Laptop.py` (thu 30 mẫu laptop/lớp).
* File dữ liệu hình ảnh `Gesture_Data.csv` chứa 63 cột tọa độ landmark chuẩn hóa và 1 cột nhãn.
* Thư mục `Voice_Data_Raw/` chứa đủ 15.000 mẫu âm thanh 1 giây, 5 lớp cân bằng 3.000 mẫu/lớp; trong mỗi lớp có 30 mẫu tự thu bằng micro laptop.
* Bảng thống kê số mẫu cho cả 2 phân hệ, xác nhận không lệch lớp quá mức trước khi chuyển sang Phase 2 (Tiền xử lý dữ liệu).
