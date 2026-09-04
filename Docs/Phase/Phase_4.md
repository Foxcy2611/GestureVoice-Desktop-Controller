# Phase 4: Đánh giá Model Real-time (Real-time Evaluation)

> Thuộc dự án **GestureVoice Desktop Controller** - Kiểm chứng 2 model đã train ở Phase 3
> (MLP gesture + DS-CNN voice) trong điều kiện sử dụng thực tế, thay vì chỉ
> tin vào số liệu trên tập test tĩnh.

## 1. Mục tiêu giai đoạn

Phase 3 chỉ đo được model học tốt đến đâu **trên chính tập dữ liệu đã thu** —
chưa chắc phản ánh đúng lúc dùng thật (webcam/mic sống, góc độ và điều kiện
môi trường không giống lúc thu mẫu). Phase 4 chạy thử 2 model qua webcam/mic
thời gian thực để trả lời 1 câu hỏi duy nhất cho mỗi model: **kết quả cao ở
Phase 3 có đáng tin khi dùng thật không, hay chỉ là con số đẹp trên giấy?**

Nguyên tắc xuyên suốt: **Phase 4 không train lại, không sửa model** — chỉ
load model đã lưu (`.keras`) và quan sát hành vi thật, ghi nhận lại để quyết
định có cần quay lại Phase 1 (thu thêm data) hoặc Phase 3 (đổi kiến
trúc/tham số) hay không.

## 2. Phạm vi công việc

* **Gesture** (`Eval_Gesture_Webcam.py`): chạy model MLP qua webcam
  real-time, thử ở điều kiện khác lúc thu (khoảng cách, góc, ánh sáng), xem
  độ chính xác/độ ổn định thực tế có khớp với 99% ở Phase 3 không.
* **Voice** (`Eval_KWS_Mic.py`): chạy model DS-CNN qua micro laptop real-time,
  thử với giọng nói/tạp âm sống thay vì file `.wav` đã cắt gọn sẵn. Theo dõi riêng các nhầm lẫn `off → up`, `on → up/down` và việc chuyển liên tiếp giữa hai lệnh.

Chi tiết quy trình từng bước cho phần gesture xem tại
`Eval_Gesture_Process.md`.

## 3. Tiêu chí đọc kết quả

| Hiện tượng quan sát | Ý nghĩa |
|---|---|
| Gesture: accuracy thực tế thấp hơn rõ rệt so với Phase 3 | Cần thu thêm dữ liệu đa dạng hơn (quay lại Phase 1) |
| Gesture: accuracy thực tế vẫn cao, ổn định | Model dùng được, đi tiếp Phase 5 |
| Voice: hay báo nhầm `Background` thành lệnh thật | Cần thêm dữ liệu `Background`, hoặc thêm ngưỡng confidence ở Phase 5 |
| Voice: nhầm giữa `Off`/`Up` hoặc giữ lệnh trước khi đổi lệnh | Kiểm tra dữ liệu micro laptop mới, confusion matrix realtime và logic bỏ phiếu/debounce; không kết luận chỉ từ accuracy offline |
| Voice: độ trễ phản hồi rõ rệt, giật | Vấn đề pipeline real-time (không phải model) — tối ưu ở tầng code |

## 4. Kết quả cuối cùng (Deliverables)

* 2 script real-time: `Eval_Gesture_Webcam.py`, `Eval_KWS_Mic.py`.
* Ghi nhận quan sát thực tế cho từng model — làm input cho Phase 5 (state
  machine cần biết ngưỡng confidence/debounce phù hợp).
* Kết luận rõ ràng: model hiện tại **đủ dùng** để đi tiếp Phase 5, hay
  **cần thu thêm/train lại** trước khi tích hợp state machine.
