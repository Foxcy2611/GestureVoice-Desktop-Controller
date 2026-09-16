# Tổ chức Dataset và Quy trình cách thu thập Dataset

Dự án **GestureVoice Desktop Controller** yêu cầu kết hợp cả hai luồng dữ liệu: Hình ảnh (nhận diện cử chỉ tay) và Âm thanh (nhận diện mệnh lệnh giọng nói). File này mô tả cách lưu trữ dữ liệu và các bước tiền xử lý trước khi đưa vào huấn luyện mô hình.

Vì lý do bản quyền, các file âm thanh thô sẽ không được tải lên Repository này.

## 1. Tổ chức thư mục

```
Dataset/
|
├── Landmarks_Normalize/    <- Tập dataset của model nhận diện cử chỉ 
|     ├── Collect_Data.md   <- File md để tự theo dõi tiến trình thu mẫu ảnh
|     └── Gesture_Data.csv  <- CSV thu thập dữ liệu, làm input cho model
|
└── Voice_Data_Raw/
|   ├── On/               <- Tập wav từ khóa "On"
|   ├── Off/              <- Tập wav từ khóa "Off"
|   ├── Up/               <- Tập wav từ khóa "Up"
|   ├── Down/             <- Tập wav từ khóa "Down"
|   ├── Background/       <- Tập wav nhiễu nền (không phải lệnh thoại)
|         ├── Kaggle/     <- Tập nhiễu được cắt từ kaggle
|         ├── INMP441/    <- Tập nhiễu được cắt từ 3 file tự thu
|         └── Laptop/     <- 30 mẫu nền 1 giây tự thu bằng micro laptop
|
└── Dataset_Kaggle        <- Tập dataset down từ Kaggle về
```

## 2. Thu thập data hình ảnh

### 2.1. Định nghĩa nhãn (labels) cố định trước khi thu

8 lớp cử chỉ, áp dụng chung cho cả tay trái và tay phải (vai trò phòng/hành động được gán ở tầng logic sau, không phải ở tầng dữ liệu):

| Nhãn | Mô tả |
|---|---|
| `fist` | Nắm tay (0 ngón) |
| `one` | Giơ 1 ngón |
| `two` | Giơ 2 ngón |
| `three` | Giơ 3 ngón |
| `four` | Giơ 4 ngón |
| `five` | Giơ 5 ngón (xòe bàn tay) |
| `like` | Biểu tượng "like" (ngón cái) |
| `ok` | Biểu tượng "ok" |

Được sử dụng theo thư viện `MediaPipe`, có thể xem qua cách sử dụng tại Repo này của Google [Import MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands.py)

> Lưu ý: danh sách này cần chốt cứng trước khi thu, tránh đổi giữa chừng làm dataset không đồng nhất.

### 2.2. Xây dựng script thu thập dữ liệu

- Bật webcam bằng OpenCV.
- Dùng MediaPipe Hands (`max_num_hands=2`) để phát hiện tay theo thời gian thực, vẽ landmark overlay lên khung hình để kiểm tra trực quan tay có đang được nhận diện đúng không.
- Với mỗi tay phát hiện được, trích xuất 21 điểm landmark (x, y, z) → tổng 63 giá trị.
- Chuẩn hóa tọa độ: quy về hệ tọa độ tương đối so với điểm cổ tay (wrist) làm gốc, và scale theo kích thước bàn tay — để model không bị ảnh hưởng bởi vị trí/khoảng cách tay trong khung hình.
- Gán nhãn bằng phím bấm: khi giữ đúng cử chỉ, bấm phím tương ứng (ví dụ `0`-`5`, `l`, `o`) để lưu mẫu hiện tại (63 giá trị: `x, y, z` + nhãn) vào file CSV.

### 2.3. Thu thập đủ số lượng và đa dạng

Để dataset đủ chất lượng, mỗi nhãn nên thu:

- Tối thiểu **150–300 mẫu/nhãn**.
- Ở **nhiều khoảng cách** khác nhau tới camera (gần, xa, trung bình).
- Ở **nhiều góc nghiêng** bàn tay khác nhau (không chỉ giơ thẳng vuông góc camera).
- Cả **tay trái và tay phải** (vì dự án dùng multi-hand).
- Nếu có thể, thu ở **vài điều kiện ánh sáng khác nhau** để tăng độ bền của model khi triển khai thực tế.

### 2.4. Kiểm tra chất lượng dữ liệu sau khi thu

- Đếm số mẫu mỗi nhãn, đảm bảo không bị lệch quá nhiều giữa các lớp (mất cân bằng dữ liệu — class imbalance — sẽ khiến model thiên vị về lớp có nhiều mẫu hơn).
- Xem nhanh vài dòng dữ liệu để đảm bảo không có giá trị lỗi (NaN, tọa độ bất thường do MediaPipe nhận diện sai/nhiễu).

### 2.5. Lưu ý khi thu thập dataset
- **Ánh sáng:** Tránh ngược sáng (ngồi quay lưng ra cửa sổ/đèn) — MediaPipe dễ mất tay hoặc landmark bị lệch khi tay bị tối/lóa. Nên có ánh sáng đều, chiếu từ phía trước hoặc bên.
- **Nền phía sau:** Nền đơn giản, ít vật thể giống da tay (không quá quan trọng với landmark-based vì MediaPipe khá bền, nhưng nền lộn xộn đôi khi gây detect nhầm hoặc mất tracking khi tay che nền phức tạp).
- **Khoảng cách tay-camera:** Nên thu ở nhiều khoảng cách (30cm–1m), tránh chỉ thu ở đúng 1 khoảng cách cố định vì khoảng cách sử dụng desktop thực tế sẽ dao động.
- **Trang phục/tay áo:** Tránh tay áo dài che cổ tay — MediaPipe cần thấy rõ điểm cổ tay (landmark 0), làm gốc chuẩn hóa.
- **Đồng nhất môi trường thu giữa các nhãn:** Nếu thu nhãn A hoàn toàn ban ngày, nhãn B hoàn toàn ban đêm → model có thể học nhầm đặc trưng ánh sáng thay vì hình dạng tay (data leakage kiểu tinh vi). Nên xen kẽ điều kiện giữa các nhãn.
- Tốc độ bấm phím: Đừng bấm liên tục quá nhanh cùng 1 tư thế — nên thay đổi góc/tư thế nhẹ giữa mỗi lần bấm để mẫu đa dạng thay vì gần như trùng lặp.
- **Cả 2 tay:** Nhớ thu đủ cả tay trái và tay phải cho từng nhãn (dù model dùng chung, nhưng hình chiếu landmark của tay trái/phải hơi khác nhau về thứ tự điểm, nên cần đại diện đủ cả hai).

---

## 3. Thu thập data âm thanh

### 3.1. Nguồn dữ liệu

Dataset KWS dùng nguồn có sẵn làm phần lớn dữ liệu, sau đó **bắt buộc bổ sung dữ liệu đúng micro laptop sẽ chạy thực tế**:

- **Google Speech Commands v2** (qua Kaggle): [Kaggle KWS](https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2?resource=download) — cung cấp sẵn các file `.wav` cho từ khóa `on`, `off`, `up`, `down`, và một tập `_background_noise_` (tạp âm nền: rửa bát, xe đạp, gõ bàn phím...).
- **Tự thu (INMP441)**: 3 file dài 5 phút, tái sử dụng từ dự án AI Asthma trước đó (im lặng, podcast, tạp âm quạt/bàn phím) — dùng làm nguồn nhiễu nền bổ sung, giúp model quen với đặc tính micro/mic thực tế thay vì chỉ nhiễu từ Kaggle.
- **Tự thu bằng micro laptop**: thu đúng **30 file `.wav` dài 1 giây cho mỗi lớp** `On`, `Off`, `Up`, `Down`, `Background` (tổng 150 file). Đây là phần dữ liệu theo đúng microphone, khoảng cách và môi trường triển khai; không được bỏ qua vì nếu chỉ train bằng nguồn ngoài, model dễ lệch miền âm thanh và đoán sai khi chạy realtime.

> Vì lý do bản quyền, các file âm thanh thô không được tải lên Repository — chỉ script tái tạo (`Collect_KWS_Data.py` và `Record_KWS_Laptop.py`) được lưu lại, người dùng tự tải Kaggle về, chạy script bốc/cắt dữ liệu, rồi tự thu 30 mẫu laptop cho từng lớp.

### 3.2. Script thu thập & cắt ghép

`Collect_KWS_Data.py` bốc/cắt phần dữ liệu nguồn; `Record_KWS_Laptop.py` thu 30 mẫu 1 giây/lớp từ micro laptop. Các biến chính của script bốc/cắt gồm `DATASET_DIR` (dataset Kaggle gốc), `OUT_DIR` (nơi ghi kết quả), số mẫu từng nguồn, `MIN_GAP_RATIO` (khoảng cách tối thiểu giữa các đoạn cắt) và `SEED` (tái lập kết quả).

Quy trình xử lý:

1. **Bốc lệnh từ Kaggle**: lấy ngẫu nhiên **2.970 file** cho mỗi lệnh `On`, `Off`, `Up`, `Down`.
2. **Cắt nền từ Kaggle và INMP441**: lấy lần lượt **1.485 + 1.485** đoạn dài 1 giây.
3. **Thu trên laptop**: chạy `Record_KWS_Laptop.py`, thu **30 file/lớp** và lưu lần lượt vào các thư mục `On/`, `Off/`, `Up/`, `Down/`, `Background/Laptop/`.
4. **Gộp nhãn**: các nguồn nền Kaggle, INMP441 và Laptop đều mang chung nhãn `background`; bài toán cuối cùng có 5 lớp (`On`, `Off`, `Up`, `Down`, `Background`).

Kết quả thô phải đúng **15.000 file wav dài 1 giây, cân bằng 3.000 file/lớp**: mỗi keyword = 2.970 Kaggle + 30 laptop; `Background` = 1.485 Kaggle + 1.485 INMP441 + 30 laptop.

### 3.3. Kiểm tra chất lượng dữ liệu sau khi thu

- Kiểm tra số lượng file thực tế mỗi lớp so với kỳ vọng (`N_COMMAND`, `N_BG_KAGGLE`, `N_BG_OWN`) — phát hiện sớm nếu nguồn Kaggle bị thiếu file cho lệnh nào đó.
- Nghe thử ngẫu nhiên một số đoạn đã cắt để đảm bảo không bị cắt giữa từ (mất phần đầu/cuối từ khóa) hoặc đoạn nhiễu bị lẫn giọng nói.
- Kiểm tra `MIN_GAP_RATIO` đã đủ để các đoạn cắt từ cùng 1 file nguồn không trùng lặp/chồng lấn quá nhiều.

---

## 4. Kết quả cuối cùng (Deliverables)

Sau khi hoàn thành Phase 1, dự án sẽ có:

1. **File dataset** (`Gesture_Data.csv`): khoảng 1000–2000+ dòng dữ liệu, mỗi dòng gồm 63 cột tọa độ landmark (đã chuẩn hóa) + 1 cột nhãn.
2. **Numpy Preprocess** (`X_voice.npy` và `Y_labels.npy`): kết quả của 15.000 file `.wav` thô KWS. Riêng 150 mẫu laptop được tạo thêm feature augmentation tĩnh, nên output hiện tại có **16.050 feature** (3.210/lớp), không phải thêm file `.wav`.

### Ví dụ cấu trúc file dataset đầu ra hình ảnh

```csv
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
five,0.52,0.61,0.00,0.55,0.58,-0.01,...,0.48,0.30,-0.05
fist,0.50,0.60,0.00,0.51,0.59,-0.01,...,0.49,0.55,-0.02
ok,0.49,0.62,0.00,0.53,0.57,-0.02,...,0.47,0.31,-0.04
```

### Ví dụ về kết quả npy dataset âm thanh

* Với `X_voice.npy` là kết tinh của trích xuất 39 hệ số (MFCC + Delta + Delta-Delta) sẽ có shape **(16050, 51, 39)** với
    * 16050: 15.000 feature gốc + 1.050 feature augmentation từ 150 mẫu laptop (mỗi mẫu laptop có 7 biến thể bổ sung)
    * 51: Số frame trên 1s
    * 39: 39 hệ số trích xuất

* Với `Y_labels.npy`, mỗi 1 cột (hay hàng ??) sẽ là đáp án của 1 mẫu 39 hệ số đặc trưng của `X_voice`.npy

|Loại âm|Tên nhãn|
|---|---|
|`On`|0|
|`Off`|1|
|`Up`|2|
|`Down`|3|
|`Background`|4|

---
