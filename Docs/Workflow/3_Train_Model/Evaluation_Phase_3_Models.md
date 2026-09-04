# Đánh giá kết quả Phase 3: MLP Gesture + DS-CNN Voice

> Tổng hợp nhận xét từ `Sumary_1.log`/`Training_History_1.png` (gesture) và `Sumary_2.log`/`Training_History_2.png` (voice/KWS). Dùng làm căn cứ quyết định đi tiếp Phase 4 hay quay lại Phase 1/3.

## 1. Tổng quan số liệu

| | Gesture (MLP) | Voice (DS-CNN) |
|---|---|---|
| Tham số | 17,866 (69.79 KB) | 18,181 (71.02 KB) |
| Epoch dừng (EarlyStopping) | 81 (best = epoch 66) | 25 (khôi phục trọng số tốt nhất ở epoch 15) |
| Test Loss | 0.0259 | 0.1128 |
| **Test Accuracy** | **99.00%** | **95.76%** |
| Số mẫu test | 300 (30/lớp × 10 lớp) | 1605 (321/lớp × 5 lớp) |

Cả 2 đều đủ tốt để đi tiếp Phase 4 — nhưng "đủ tốt" ở đây nghĩa là **không có lỗi kỹ thuật rõ ràng trong quá trình train**, chưa phải là xác nhận khả năng tổng quát hóa thật (đó là việc của Phase 4).

## 2. Gesture (MLP) — chi tiết

### Đọc đồ thị train (`Training_History_1.png`)

* **Val Accuracy cao hơn Train Accuracy suốt quá trình** — không phải dấu hiệu leakage, mà là artefact của `Dropout(0.3)`: lúc train, 30% neuron bị tắt ngẫu nhiên mỗi lần forward khiến train accuracy bị "thiệt" so với thực lực; lúc validate, Keras tự tắt Dropout (dùng full network) nên val accuracy có lợi thế hơn. Pattern này bình thường với model có Dropout mạnh.
* Cả 2 đường hội tụ mượt, không dao động mạnh, không có dấu hiệu val loss tăng trở lại (dấu hiệu overfit rõ) — quá trình train ổn định.

### Đọc confusion matrix

* Chỉ 3/300 mẫu sai, rải rác ở 3 cặp khác nhau (`left_two`↔`right_three`, `right_fist`↔`right_like`, `right_ok`↔`right_five`) — không tập trung vào 1 cặp cụ thể, giống nhiễu ngẫu nhiên hơn là lỗi hệ thống hoặc 2 gesture bị nhầm hình học lặp lại.
* 99% với n=300 có sai số thống kê tự nhiên cỡ ±1.1% (95% CI ≈ 96.8%–100%) — không nên coi đây là con số "chính xác tuyệt đối".

### Nghi vấn cần Phase 4 xác nhận

Dữ liệu gesture thu bằng bấm phím liên tục cùng 1 tư thế (README mục 2.5 đã cảnh báo rủi ro này) — `train_test_split` chia ngẫu nhiên theo dòng, không theo cụm tư thế, nên các mẫu gần-trùng-lặp hoàn toàn có thể vừa nằm ở train vừa nằm ở test. 99% một phần có thể phản ánh độ trùng lặp dữ liệu chứ chưa hẳn khả năng tổng quát hóa thật.

→ **Việc cần làm ở Phase 4**: test qua webcam ở góc/khoảng cách/ánh sáng **chưa từng thu**. Nếu accuracy thực tế tụt rõ so với 99%, xác nhận nghi vấn leakage — cần quay lại Phase 1 thu thêm dữ liệu đa dạng hơn thay vì đổi kiến trúc.

## 3. Voice (DS-CNN) — chi tiết

### Nguồn đánh giá

Các số liệu ở phần này lấy trực tiếp từ `3_Train_Model/Output_KWS/Sumary_2.log` và `Training_History_2.png`, sau khi tiền xử lý bằng `librosa.load`. Input là `(16050, 51, 39, 1)`, cân bằng 3.210 feature/lớp; train/validation/test lần lượt là 12.840 / 1.605 / 1.605 mẫu.

### Đọc đồ thị train (`Training_History_2.png`)

* Validation dao động khá mạnh ở giai đoạn đầu, rồi ổn định hơn từ khoảng epoch 14. Điều này phù hợp với dataset KWS có các lớp âm học gần nhau, không tự nó chứng minh lỗi pipeline.
* `EarlyStopping` dừng ở epoch 25 và khôi phục trọng số tốt nhất tại epoch 15. Test loss 0.1128 và test accuracy 95.76% cho thấy model học được bài toán, nhưng không nên diễn giải là accuracy deploy tuyệt đối.

### Đọc confusion matrix

```
          on  off   up  down  bg
on       303    3    9     5   1
off       10  284   25     1   1
up         1    2  314     3   1
down       3    0    3   315   0
background 0    0    0     0 321
```

* Sai nhiều nhất là **`off → up` (25/321)**; vì vậy `up` có recall cao (97,82%) nhưng precision thấp hơn (89,46%). `off` có recall thấp nhất (88,47%). `on → up` (9) và `on → down` (5) là các nhầm lẫn đáng theo dõi tiếp.
* `background` có recall 100% trong test split, nhưng không được xem là bảo đảm tuyệt đối khi deploy: các đoạn nền có thể xuất phát từ cùng file nguồn.
* 150 mẫu laptop được tạo thêm 7 biến thể/feature trước khi split. Vì feature cùng một file laptop gốc có thể nằm ở nhiều split, con số 95,76% có khả năng lạc quan. Đây là giới hạn của cách giữ nguyên script train hiện tại, không phải bằng chứng pipeline sai.

→ **Việc cần làm ở Phase 4**: test bằng micro laptop với câu lệnh/lần phát âm và tạp âm thật chưa từng dùng trong train; ghi riêng confusion cho `off ↔ up`, `on → up/down` và background. Bộ đánh giá realtime cũng cần bỏ phiếu theo chuỗi dự đoán mới để tránh trạng thái của lệnh trước làm nhiễu lệnh kế tiếp.

## 4. So sánh 2 model

* Quy mô tham số gần tương đương (~17.9K vs ~18.2K) — không phải yếu tố tạo ra chênh lệch accuracy (99% vs 95.76%).
* Chênh lệch accuracy phần lớn đến từ **độ khó bài toán khác nhau**: gesture là input đã feature-engineer sẵn, ranh giới phân loại rõ ràng hơn; voice phải học từ tín hiệu âm thanh thô hơn (dù đã qua MFCC), biến thiên tự nhiên giữa các lần phát âm lớn hơn biến thiên hình học giữa các lần giơ tay.
* Cả 2 đều có **cùng 1 dạng rủi ro**: nghi vấn near-duplicate giữa train/test do cách thu dữ liệu (bấm phím liên tục / cắt nhiều đoạn từ ít file nguồn) — đây là điểm chung cần Phase 4 xác nhận cho cả 2 model, không riêng gì 1 bên.

## 5. Kết luận & bước tiếp theo

| Model | Sẵn sàng Phase 4? | Việc cần làm cụ thể |
|---|---|---|
| Gesture MLP | ✅ | Test webcam ở điều kiện chưa từng thu; đối chiếu lại nếu accuracy tụt rõ so với 99% |
| Voice DS-CNN | ✅ có điều kiện | Test micro laptop với dữ liệu chưa từng thu; theo dõi `off → up`, `on → up/down` và tránh bỏ phiếu lẫn trạng thái lệnh trước |

Không cần train lại hay đổi kiến trúc ở thời điểm này — cả 2 model đủ điều kiện đi tiếp Phase 4. Quyết định có quay lại Phase 1 (thu thêm) hay Phase 3 (đổi tham số) sẽ dựa trên kết quả thực tế đo được ở Phase 4, không suy đoán trước.
