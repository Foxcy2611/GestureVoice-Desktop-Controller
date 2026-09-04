# Quy trình Eval Model Gesture Real-time

> Chi tiết kỹ thuật cho phần gesture của Phase 4. Xem `Phase_4.md` cho bức
> tranh tổng quát của cả giai đoạn.

## 1. Vì sao cần bước này

Kết quả 99% test accuracy ở `Sumary_1.log` (Phase 3) đo trên tập test được
tách ra từ **cùng 1 phiên thu dữ liệu** với tập train. README Phase 1 (mục
2.5) từng cảnh báo rủi ro: nếu lúc thu bấm phím liên tục quá nhanh cùng 1 tư
thế, các mẫu train/test có thể gần như trùng nhau (near-duplicate) — khiến
99% là con số "học thuộc lòng" chứ không phải tổng quát hóa thật.

Test qua webcam ở điều kiện **chưa từng xuất hiện lúc thu** (góc mới,
khoảng cách mới, ánh sáng mới) là cách duy nhất trả lời dứt điểm nghi vấn
này — số liệu tập test tĩnh không làm được, vì tập test vẫn được rút ra từ
cùng phân bố dữ liệu với tập train.

## 2. Chuẩn bị trước khi chạy

* File `Gestures_MLP_Model.keras` (hoặc `Gestures_MLP_Best.keras`) đã có ở
  `3_Train_Model/Output_Gestures/`.
* File `Z_classes.npy` đã có ở `2_Data_Preprocessing/Output_npy/`.
* Script `Eval_Gesture_Webcam.py` — đối chiếu lại hàm `Normalize_Landmarks()`
  với `Collect_Image_Data.py` **trước khi chạy**, đảm bảo công thức chuẩn
  hóa khớp tuyệt đối (khác công thức → model không lỗi nhưng dự đoán sai
  loạn, rất khó nhận ra nếu không kiểm tra kỹ).
* Webcam hoạt động, đủ ánh sáng để MediaPipe track ổn định.

## 3. Các bước thực hiện

### Bước 1 — Chạy script, xác nhận load đúng

```bash
python Eval_Gesture_Webcam.py
```

Kiểm tra log in ra: đúng đường dẫn model, đúng danh sách 10 nhãn
(`left_one`, `left_two`, `right_fist`, ...).

### Bước 2 — Test ở điều kiện GIỐNG lúc thu (sanity check)

Trước khi test biến thể, thử vài gesture cơ bản đúng kiểu lúc thu (khoảng
cách/góc quen thuộc) để chắc pipeline hoạt động đúng — nếu ngay bước này đã
sai, lỗi nằm ở code (path, thứ tự nhãn, công thức chuẩn hóa lệch), không
phải do model kém.

### Bước 3 — Test có chủ đích ở điều kiện KHÁC lúc thu

Đổi từng biến số một lượt, quan sát và ghi nhận riêng cho từng cái:

| Biến số | Cách test |
|---|---|
| Khoảng cách | Gần hơn (~20cm), xa hơn (~1.2m) so với lúc thu |
| Góc nghiêng | Góc chéo gắt hơn mức đã thu (nhưng vẫn trong ngưỡng MediaPipe track được) |
| Ánh sáng | Đổi hẳn điều kiện sáng (nếu lúc thu là đèn LED, test dưới ánh sáng tự nhiên) |
| Tay di chuyển | Giữ gesture trong lúc tay đang di chuyển nhẹ, không đứng yên tuyệt đối như lúc thu |
| Người khác (nếu có) | Người khác thử — kiểm tra model có overfit vào đặc điểm tay của riêng bạn không |

### Bước 4 — Đối chiếu Left/Right hiển thị song song

Script hiển thị cả nhãn MediaPipe nhận diện (`[Left]`/`[Right]`) cạnh nhãn
model dự đoán. Nếu MediaPipe báo `Left` nhưng model dự đoán `right_xxx` (chữ
đỏ cảnh báo lệch), đó là dấu hiệu model đang nhầm — ghi nhận lại, không bỏ
qua dù confidence cao.

### Bước 5 — Test riêng giới hạn thiết kế (tay trái ngoài 2 case đã học)

Tay trái trong lúc thu **chỉ có 2 case** (`one`, `two`). Chủ động giơ tay
trái làm `three`/`four`/`five` để xem model phản ứng ra sao (sẽ luôn ép về
1 trong 10 nhãn đã biết, không có nhãn "không xác định") — ghi nhận hành vi
này, đây là giới hạn thiết kế đã biết trước, không phải bug.

**Kết quả thực tế đã kiểm chứng:** giơ tay trái làm `three` (case tay trái
không hề học), model dự đoán ra `right_three` — đúng như dự đoán, vì hình
dạng landmark của `three` giống case tay phải `right_three` hơn là 2 case
tay trái đã học (`one`/`two`), nên softmax bị kéo về đó thay vì ép nhầm
thành `left_one`/`left_two`. Quan trọng hơn: MediaPipe vẫn báo đúng tay là
`Left`, nên cơ chế cảnh báo lệch (Bước 4) bắt được ngay — hiển thị chữ đỏ
`right_three [Left]` — xác nhận lớp bảo vệ này **hoạt động đúng như thiết
kế**: không ngăn được model dự đoán sai (vì nằm ngoài phạm vi đã học), nhưng
phát hiện và cảnh báo được ngay khi nó xảy ra, đủ để tầng logic phía sau
(Phase 5) biết mà bỏ qua kết quả này thay vì thực thi nhầm hành động.

## 4. Ghi nhận kết quả

Với mỗi biến số ở Bước 3, ghi lại:

* Gesture nào bị nhận sai / không ổn định (dự đoán nhảy qua lại giữa các
  nhãn dù tay đứng yên).
* Confidence dao động trong khoảng nào khi tay di chuyển (so với lúc đứng
  yên hoàn toàn).
* Có xảy ra cảnh báo lệch Left/Right (Bước 4) không, ở gesture/điều kiện
  nào.

## 5. Kết luận

**Kết quả thực tế đã kiểm chứng:**

* Đã test ở nhiều điều kiện ánh sáng khác nhau (ánh sáng mặt trời trực
  tiếp, phòng tối) — tay phải nhận đúng đủ các trường hợp, độ tự tin cao
  (>96-97%) ở mọi điều kiện.
* Tay trái nhận đúng 2 case đã học (`left_one`, `left_two`).
* Giơ tay trái ngoài phạm vi đã học (VD `three`) → model dự đoán nhầm sang
  `right_three` như dự đoán, nhưng cơ chế cảnh báo lệch Left/Right (Bước 4)
  bắt được ngay — xác nhận lớp bảo vệ này hoạt động đúng thiết kế.
* Đã test trên **tay của người khác** (không phải người thu dữ liệu) —
  kết quả vẫn ổn, không có dấu hiệu overfit vào đặc điểm tay riêng của
  người thu.
* Đã test thay đổi **khoảng cách gần/xa** — kết quả vẫn ổn định.

**Kết luận:** nghi vấn near-duplicate ở mục 1 được giải tỏa — 99% ở Phase 3
là con số phản ánh đúng khả năng tổng quát hóa thật của model, không phải
"học thuộc lòng". Model **đủ điều kiện đi tiếp Phase 5** (state machine),
không cần quay lại Phase 1 thu thêm dữ liệu hay Phase 3 đổi kiến trúc.