# Huấn luyện Model Voice (DS-CNN)

> Chi tiết kiến trúc, lý do lựa chọn, và quy trình huấn luyện model phân loại lệnh thoại từ đặc trưng MFCC. Thuộc Phase 3 của dự án **GestureVoice Desktop Controller**.

## 1. Input

| File | Shape | Mô tả |
|---|---|---|
| `X_voice.npy` | `(N, T, 39)` | MFCC (13) + delta (13) + delta-delta (13) theo `T` khung thời gian, trích từ Phase 2 |
| `Y_labels.npy` | `(N,)` | Nhãn cố định: 0=On, 1=Off, 2=Up, 3=Down, 4=Background |

> `Conv2D`/`SeparableConv2D` trong Keras luôn đòi input rank 4: `(batch, height, width, channels)`. `X_voice.npy` chỉ có rank 3 `(N, T, 39)` — thiếu trục channel. Trước khi đưa vào model cần thêm 1 trục dummy: `X = X[..., np.newaxis]` → `(N, T, 39, 1)`, coi khung MFCC×frame như 1 "ảnh xám" 1 kênh. Bước này không thêm thông tin gì, chỉ đổi hình dạng tensor cho đúng yêu cầu của layer.

## 2. Vì sao chọn DS-CNN (Depthwise Separable CNN)

Khác với gesture, input voice **có cấu trúc tuần tự thời gian thật sự** (giá trị MFCC ở khung thời gian gần nhau có liên hệ) — đúng dạng bài toán Keyword Spotting (KWS) kinh điển, nên kiến trúc CNN 2D áp trên "ảnh" (thời gian × hệ số MFCC) là lựa chọn phù hợp về mặt bài toán.

Về việc chọn cụ thể DS-CNN thay vì CNN thường/CRNN: theo Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2018), *"Hello Edge: Keyword Spotting on Microcontrollers"*, arXiv:1711.07128 — <cite index="17-1">nhóm tác giả so sánh nhiều kiến trúc (DNN, CNN, RNN, CRNN, DS-CNN) trên cùng bài toán KWS và cho thấy DS-CNN đạt độ chính xác tốt nhất, tận dụng các lớp depthwise separable convolution vốn ít tốn tài nguyên tính toán/bộ nhớ hơn convolution thường.</cite> Điểm quan trọng: DS-CNN đạt accuracy cao nhất **không phải vì được đo dưới ràng buộc MCU** mà là do depthwise separable convolution factorize phép tích chập thành 2 bước nhỏ hơn (depthwise + pointwise), giúp mạng có thể xếp **nhiều lớp hơn** ở cùng ngân sách tham số so với CNN thường — tức là nó vẫn là lựa chọn tốt về accuracy thuần túy, không chỉ vì nhẹ. Dự án cũng đã dùng kiến trúc này ở AI Asthma trước đó nên tái sử dụng được kinh nghiệm/pipeline.

## 3. Kiến trúc (chốt)

Input dạng "ảnh 1 kênh" `(T, 39, 1)` — trục thời gian × trục hệ số MFCC, tương tự cách Hello Edge coi đặc trưng âm thanh như ảnh 2D để đưa vào CNN:

`Input(T,39,1) → Conv2D(64, kernel=10×4, stride=2) + BatchNorm + ReLU → [SeparableConv2D(64, kernel=3×3) + BatchNorm + ReLU] × 3 khối → GlobalAveragePooling2D → Dropout(0.4) → Dense(5) + Softmax`

* **`SeparableConv2D`**: đây chính là API Keras triển khai depthwise separable convolution mô tả trong Hello Edge — không có class riêng tên "DepthwiseSeparableConv2D", `SeparableConv2D` đã gộp sẵn 2 bước (depthwise + pointwise) trong 1 lệnh gọi.
* **Vì sao có `Conv2D` mở đầu thay vì dùng `SeparableConv2D` xuyên suốt**: đây là pattern kế thừa từ chính MobileNet — kiến trúc mà Hello Edge dựa vào để xây DS-CNN. Lớp đầu tiên của MobileNet là 1 convolution thường ("full convolution"), tất cả các lớp sau đó mới là depthwise separable. Lý do: ở lớp đầu, input chỉ có 1 kênh (ảnh xám/khung MFCC) — bước "depthwise" gần như không có gì để tách riêng khi chỉ có 1 kênh, nên lợi ích tiết kiệm tham số của `SeparableConv2D` ở ngay lớp đầu rất nhỏ, trong khi 1 `Conv2D` thường với kernel lớn hơn học đặc trưng thô ban đầu linh hoạt hơn. Lợi ích thật sự của `SeparableConv2D` chỉ phát huy rõ từ lớp thứ 2 trở đi, khi số kênh đã tăng lên (64, 128...).
* **Kernel `10×4` (không vuông 3×3)**: đây là điểm cần nói rõ — con số `10×4` cụ thể không phải trích nguyên văn từ Hello Edge (paper không công bố 1 con số cố định, các biến thể model trong đó khác nhau), mà là áp dụng **nguyên tắc kernel bất đối xứng thời gian/tần số** phổ biến trong literature CNN cho speech: Sainath & Parada (2015), *"Convolutional Neural Networks for Small-footprint Keyword Spotting"* — chính là paper Hello Edge trích dẫn khi so sánh baseline CNN — dùng kernel rộng theo trục thời gian (bằng cả chiều thời gian của input) và hẹp hơn theo trục tần số (kernel cao 8, stride dọc 4). Lý do kỹ thuật: 2 trục trong input MFCC×thời gian mang ý nghĩa vật lý khác nhau hoàn toàn (không giống ảnh chụp, nơi 2 trục không gian đối xứng về ý nghĩa) — trục thời gian cần "nhìn" xa hơn để bắt trọn diễn biến của 1 âm tiết, trục hệ số MFCC không có tính liên tục không gian như pixel ảnh. Ép kernel vuông 3×3 (như hướng AI Asthma trước đây) vẫn chạy được và không sai, chỉ là không tận dụng đặc thù bất đối xứng này — 2 hướng đều có cơ sở, đây là lựa chọn thiết kế chứ không phải đúng/sai tuyệt đối.
* **3 khối `SeparableConv2D`, chưa thêm khối thứ 4**: bài toán chỉ 5 lớp trên clip 1 giây là nhỏ hơn nhiều so với bài toán gốc trong Hello Edge (nhiều từ khóa hơn) — 3 khối là điểm khởi đầu hợp lý. Chỉ thêm khối thứ 4 nếu sau khi train thấy dấu hiệu underfit (train accuracy cũng thấp).
* **`GlobalAveragePooling2D`**: thay vì `Flatten` (gộp hết giá trị còn lại của feature map thành 1 vector dài rồi qua Dense lớn — rất nhiều tham số, dễ overfit), GAP lấy **trung bình toàn bộ giá trị của từng feature map** thành 1 con số/kênh (VD 64 feature map cuối → ra đúng vector 64 số). Vừa giảm mạnh tham số ở phần đầu ra, vừa không phụ thuộc vị trí chính xác trong thời gian (từ khóa nói hơi sớm/muộn trong clip 1 giây vẫn nhận ra được).
* **Dropout(0.4)** trước lớp output: nhiễu nền (`Background`) từ 2 nguồn khác nhau (Kaggle + INMP441) có đặc tính hơi khác nhau, dropout giúp model không học lệch theo đặc trưng riêng của 1 nguồn nhiễu.

## 4. Cấu hình huấn luyện (chốt)

* **Optimizer**: Adam, loss `sparse_categorical_crossentropy`.
* **Chia tập**: 80/10/10 (train/val/test), `stratify` theo nhãn.
* **Callbacks sử dụng**:
  * `EarlyStopping` (theo dõi `val_loss`) — dừng sớm khi val_loss không cải thiện.
  * `ModelCheckpoint` — ghi checkpoint tốt nhất ra đĩa theo từng epoch, quan trọng hơn với model này vì thời gian train lâu hơn MLP nhiều (16.050 feature, nhiều lớp conv).
  * `CSVLogger` — ghi log loss/accuracy theo epoch, dùng vẽ learning curve cho báo cáo.
  * `TerminateOnNaN` — dừng ngay nếu loss "nổ" thành NaN; DS-CNN nhiều lớp + BatchNorm nên rủi ro này cao hơn MLP nếu LR hơi cao.
  * `ReduceLROnPlateau` — giảm learning rate khi `val_loss` chững lại, giúp "vắt" thêm accuracy ở các epoch cuối mà không cần train lại từ đầu với LR khác — MLP không cần nhưng DS-CNN nhiều lớp hơn nên dễ chững, đáng có.
  * **Không dùng**: `LearningRateScheduler` (trùng vai trò với `ReduceLROnPlateau`, tránh xung đột logic), `TensorBoard` (CSVLogger + confusion matrix đã đủ debug).
  * `BackupAndRestore`: tùy chọn — chỉ bật nếu thời gian train thực tế đủ dài (hàng chục phút trở lên trên laptop CPU) và có rủi ro bị gián đoạn giữa chừng.
* **Class weight**: không bật mặc định — output hiện tại cân bằng 3.210 feature/lớp (3.000 file thô/lớp, trong đó 30 file laptop được augmentation) từ `Collect_KWS_Data.py` + `Record_KWS_Laptop.py`. Chỉ bật nếu kiểm tra `np.bincount(y)` cho thấy lệch do thiếu file nguồn hoặc dữ liệu thu không đủ.

## 5. Đọc kết quả đánh giá

* **`Background` bị nhầm với `On`/`Off`**: nghi ngờ do 1 trong 2 nguồn nhiễu (Kaggle hoặc INMP441) không đủ đa dạng — xem lại confusion matrix để biết bị nhầm với lớp nào nhiều nhất, kiểm tra thử vài đoạn nhiễu bị nhận nhầm.
* **`Up`/`Down` bị nhầm lẫn nhau nhiều hơn `On`/`Off`**: hai từ có thể gần giống nhau về đặc trưng âm học hơn cặp còn lại (thực tế thường gặp trong KWS) — không phải lỗi tiền xử lý, nếu accuracy tổng vẫn ổn thì chấp nhận được, business rule ở Phase 5 (state machine) có thể thêm ngưỡng confidence cao hơn riêng cho 2 lệnh này.
* **Overfit (train cao, val/test thấp)**: tăng Dropout, giảm số khối `SeparableConv2D`, hoặc kiểm tra lại `MIN_GAP_RATIO` ở Phase 2 — nếu các đoạn cắt từ cùng 1 file nguồn quá giống nhau, val/test set có thể bị rò rỉ thông tin từ train set (leakage theo file nguồn).
* **Val_loss chững sớm dù chưa overfit**: kiểm tra log của `ReduceLROnPlateau` trong file `CSVLogger` — nếu learning rate đã giảm nhiều lần mà vẫn chững, đó là dấu hiệu model đã tới giới hạn với kiến trúc/dữ liệu hiện tại, không phải do LR.

## 6. Deliverables

* `KWS_DS_CNN_Model.keras` (và checkpoint `KWS_DS_CNN_Best.keras`): model đã train, sẵn sàng nạp lại ở Phase 4 (`Eval_KWS_Mic.py`).
* Báo cáo `classification_report` + confusion matrix trên tập test.
* Bảng ánh xạ nhãn cố định (`CLASS_NAMES` trong script) — không cần file `.npy` riêng vì đã fix cứng, chỉ cần giữ đồng bộ giữa script train và script eval real-time.
