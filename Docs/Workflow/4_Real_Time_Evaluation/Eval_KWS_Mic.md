# Quy trình đánh giá KWS qua micro laptop real-time

> Tài liệu Phase 4 cho model DS-CNN Keyword Spotting (KWS). Script chính thức là `Eval_KWS_Mic.py` (đổi tên từ bản thử nghiệm `Eval_KWS_Fix.py`). Mục tiêu là xác nhận model đã train có nhận đúng `on`, `off`, `up`, `down` qua đúng micro laptop hay không; không chỉ dựa vào test offline.

## 1. Vì sao cần đánh giá real-time

`Sumary_2.log` cho kết quả test offline **95,76%** (1.605 mẫu, 321 mẫu/lớp), nhưng đây chưa phải kết quả deploy. Dữ liệu feature đã được augmentation trước khi split nên các biến thể của cùng một file `laptop_*.wav` có thể nằm ở nhiều split. Ngoài ra, microphone laptop có noise floor, AGC, khoảng cách nói và sample rate thực tế khác nguồn Kaggle/INMP441.

Các nhầm lẫn offline cần theo dõi khi test mic:

| Nhầm lẫn | Số mẫu trong test | Ý nghĩa khi test thực tế |
|---|---:|---|
| `off → up` | 25 / 321 | Rủi ro lớn nhất; kiểm tra kỹ khi chuyển qua lại hai lệnh |
| `on → up` | 9 / 321 | Theo dõi ở giọng nói nhỏ/nhanh |
| `on → down` | 5 / 321 | Theo dõi ở giọng nói nhỏ/nhanh |
| `background` | 0 false negative | Không được hiểu là miễn nhiễu tuyệt đối; vẫn phải test tiếng nền lạ |

Vì vậy, kết quả Phase 4 là căn cứ quyết định giữ model hiện tại hay cần thu thêm dữ liệu/training lại.

## 2. Chuẩn bị trước khi chạy

* Đã có `3_Train_Model/Output_KWS/KWS_DS_CNN_Model.keras`.
* Đã dùng đúng pipeline train `Preprocess_KWS_Raw.py`: audio 16 kHz, 1 giây, MFCC 13 + delta + delta-delta, feature `(51, 39)`.
* Có micro laptop hoạt động. Script ưu tiên endpoint Windows WASAPI của micro mặc định khi có thể.
* Khi script calibrate 2 giây đầu, giữ yên lặng tương đối. Không nói hoặc gõ bàn phím trong giai đoạn này vì noise floor quyết định energy gate.

> Không đổi `CONFIDENCE_THRESHOLD` hoặc các ngưỡng RMS trước khi có kết quả baseline. Gate hiện tại tự hiệu chỉnh theo mic; chỉnh ngưỡng sớm sẽ làm khó phân biệt lỗi model với lỗi cấu hình.

## 3. Các bước thực hiện

### Bước 1 — Chạy và kiểm tra cấu hình

Từ thư mục gốc project:

```bash
python 4_Real_Time_Evaluation/Eval_KWS_Mic.py
```

Kiểm tra log đầu chương trình có:

* Model `KWS_DS_CNN_Model.keras` và danh sách lớp `on`, `off`, `up`, `down`, `background`.
* Đúng microphone đầu vào và sample rate gốc; script sẽ resample về 16 kHz khi cần.
* `Noise RMS`, `Energy gate` và `VAD threshold` được in sau 2 giây calibration.
* Chế độ log mặc định là gọn, chỉ in keyword đã xác nhận. Chỉ đổi `VERBOSE_LOGGING = True` khi cần chẩn đoán RMS, top probabilities hoặc voting.

### Bước 2 — Baseline ở điều kiện bình thường

Ngồi ở khoảng cách dùng thật, nói giọng bình thường, mỗi lệnh `on`, `off`, `up`, `down` ít nhất 20 lần. Giữa hai lần nói chờ script xử lý xong và cooldown kết thúc.

Một lệnh được tính đúng khi log in ra đúng keyword đã nói. Ghi nhận riêng ba dạng lỗi: không nhận, nhận nhầm keyword, hoặc nhận chậm/lặp lệnh.

### Bước 3 — Test chuyển lệnh liên tiếp

Lần lượt lặp các chuỗi sau ít nhất 20 chuỗi/mẫu:

```text
on  → off → on  → off
up  → down → up → down
```

Đây là test bắt buộc vì phiên bản trước từng bị phiếu cũ của lệnh trước ảnh hưởng sang lệnh sau. `Eval_KWS_Mic.py` xóa lịch sử voting nếu nhãn thô đổi và bỏ qua toàn bộ cửa sổ trong cooldown, nên một lệnh mới cần có các dự đoán liên tiếp của chính nó trước khi được xác nhận.

### Bước 4 — Test độ bền môi trường

Thay đổi từng điều kiện một, vẫn ghi riêng kết quả của từng keyword:

| Điều kiện | Cách test |
|---|---|
| Âm lượng | Nói nhỏ vừa đủ nghe, giọng bình thường và hơi to; không cần ghé sát mic |
| Khoảng cách | Gần, khoảng cách dùng thường ngày, và xa hơn một chút |
| Tiếng nền | Quạt laptop, gõ phím nhẹ, người nói ở xa hoặc âm thanh sinh hoạt |
| Không có lệnh | Im lặng, tiếng quạt và gõ phím trong 1–2 phút; không được tự in command |
| Từ ngoài tập | Nói một từ khác với `on/off/up/down`; kiểm tra model không kích hoạt liên tục |

Nếu có tiếng nền lớn hoặc background bị nhận thành lệnh, giữ lại thời điểm/điều kiện xảy ra. Không nên vội giảm energy gate: cần xem `VERBOSE_LOGGING` để biết cửa sổ đã qua gate hay model đã phân loại sai.

## 4. Cách đọc log và cơ chế bảo vệ

Luồng xử lý là: thu cửa sổ 1,5 giây → energy gate theo RMS cực đại của cửa sổ con 50 ms → VAD lấy cụm giọng nói → resample 16 kHz → giảm noise ổn định → trim/căn giữa 1 giây → peak-normalize → MFCC + delta + delta-delta → model → voting.

Một keyword chỉ được in khi đồng thời thỏa các điều kiện sau:

1. Âm thanh qua energy gate tự hiệu chỉnh.
2. Lớp dự đoán không phải `background`.
3. Có ít nhất 2 dự đoán liên tiếp cùng nhãn trong tối đa 4 cửa sổ.
4. Confidence trung bình của các phiếu đạt ít nhất 0,70.
5. Không còn trong cooldown 1,5 giây của command trước.

Thiết kế này chủ đích giảm false trigger và nhầm lẫn do cửa sổ vẫn chứa âm thanh của command cũ. Đổi lại, command có độ trễ ngắn và mỗi lần phát âm chỉ in một lần.

## 5. Ghi nhận kết quả

Sau các bước test, chỉ cần ghi ngắn gọn các điều kiện đã thử, keyword còn nhầm hoặc không nhận, số false trigger nếu có và việc chuyển liên tiếp `on/off`, `up/down` có còn giữ lệnh trước hay không. Khi cần chẩn đoán một lỗi cụ thể, bật `VERBOSE_LOGGING` và ghi lại RMS, gate, top probabilities và trạng thái voting tại thời điểm lỗi.

## 6. Kết luận Phase 4

Model KWS hiện tại đã đạt kết quả offline 95,76%, nhận diện mic laptop gần đúng ở giọng bình thường, không yêu cầu nói quá to, và lỗi giữ nhãn của command trước đã được xử lý trong script chính thức. Sau khi xác nhận lại các test ở trên, kết luận Phase 4 là model **đủ điều kiện chuyển sang Phase 5** để tích hợp state machine và điều khiển thiết bị.

Các nhầm lẫn còn lại như `off → up` được xem là giới hạn cần tiếp tục theo dõi khi tích hợp, không chặn việc đi tiếp Phase 5. Chỉ quay lại thu thêm/train lại nếu khi dùng thực tế phát hiện lỗi lặp lại có hệ thống hoặc false trigger ảnh hưởng trực tiếp đến thao tác điều khiển.
