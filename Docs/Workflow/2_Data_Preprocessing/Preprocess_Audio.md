# Tiền xử lý cho model âm thanh

## Các nguồn báo
* Warden, P. (2018). *"Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"* — phương pháp xử lý tập `background/silence`.

* Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). *"Hello Edge: Keyword Spotting on Microcontrollers"* — phương pháp trích đặc trưng MFCC cho tập từ khóa lệnh.

## Warden, P. (2018)

### Tổ chức File
* Không cắt ngắn: Khác với các file từ khóa bị ép chặt 1s, thư mục `_background_noise_` chứa các file dài vài phút (16kHz)
* Chứa các tạp âm liên tục

### Kỹ thuật được đề cập
* **Warden** không tiền xử lý chết (hard-code) nhiễu vào file, mà thực hiện động (dynamic) trong mỗi epoch huấn luyện qua 2 kỹ thuật sau phục vụ cho 2 mục đích khác nhau:
    * Kỹ thuật 1: Sinh nhãn `_silence_` (Im lặng)
        * Code huấn luyện sẽ tự động cắt ngẫu nhiên các đoạn đúng 1 giây từ các file âm thanh dài trong thư mục `_background_noise_`.
        * Các đoạn này được gán nhãn là `_silence_` (hoặc negative class). Dạy cho AI biết: "Đây là tiếng ồn, không phải lệnh, hãy bỏ qua".

    * Kỹ thuật 2: Trộn nhiễu nền (Background Mixing / Augmentation)
        * Khi load một file từ khóa (VD: "yes"), hệ thống đồng thời "bốc" ngẫu nhiên 1 giây từ thư mục noise.
        * Nhân đoạn noise này với một hệ số âm lượng ngẫu nhiên (thường < 10% âm lượng gốc để không lấn át từ khóa).
        * Cộng chồng (Mix) mảng tín hiệu âm thanh của noise vào mảng tín hiệu của từ khóa.
        * **Kết quả:** Mỗi lần file "yes" được đưa vào mạng nơ-ron, nó lại mang một "lớp nền" tạp âm khác nhau, giúp model cực kỳ trâu bò (robust) với nhiễu.

## "Hello Edge" (Zhang et al., 2017)

## Quy trình tiền xử lý

* Paper này định hình các thông số trích xuất đặc trưng kinh điển cho vi điều khiển, tối ưu để vừa giữ được thông tin, vừa không làm phình ma trận đầu vào:
    * **Đầu vào:** Âm thanh chuẩn hóa 16kHz, độ dài 1 giây.
    * **Cắt khung (Framing / Windowing):** Dùng cửa sổ 30ms (hoặc 40ms) và bước trượt (stride/hop) 10ms (hoặc 20ms) để trượt qua 1 giây âm thanh. Quá trình này tạo ra một chuỗi khoảng 49 đến 98 khung thời gian.
    * **Trích xuất đặc trưng (Feature Extraction):** Tùy vào loại model mà họ chọn số lượng hệ số khác nhau:
        * **MFCC (Mel-Frequency Cepstral Coefficients):** Dùng 10 hệ số. Lọc rất mạnh, cực kỳ tiết kiệm không gian bộ nhớ. Thường dùng cho các mạng DNN cơ bản.
        * **LFBE (Log-Mel Filterbank Energies):** Dùng 40 dải lọc. Dữ liệu gốc giữ được nhiều chi tiết phổ hơn. Phù hợp làm ma trận ảnh 2D đầu vào cho các mạng CNN.

### Đột phá về Kiến trúc Mô hình (DS-CNN)

* Bài báo thử nghiệm nhiều kiến trúc ``(DNN, CNN, RNN, CRNN)``, nhưng phát hiện lớn nhất và mang tính chuẩn mực là việc ứng dụng **DS-CNN (Depthwise Separable CNN)** cho KWS trên vi điều khiển.

* **Cơ chế:** Tách phép tích chập (Convolution) chuẩn thành hai bước tính toán riêng biệt: Depthwise (tích chập theo từng kênh) và Pointwise (tích chập 1x1 để gom kênh).

* **Lý do chọn:** Việc phân tách này giúp giảm số lượng tham số (weights) và số phép nhân cộng (MACs) đi rất nhiều lần so với CNN truyền thống, trong khi độ chính xác chỉ suy giảm vô cùng nhỏ. Đây chính là "vũ khí" giúp mô hình nhận diện âm thanh chạy mượt trên các dòng chip tài nguyên thấp mà không bị tràn RAM.


## Lưu ý về các kiến trúc mạng lựa chọn

* Theo Zhang, kiến trúc DS-CNN là mạng neuron ngon nhất với tỉ lệ chính xác cao (Có thể đọc ở README.md của repo họ để biết thêm tại đây [Repo Hello Edge Zhang](https://github.com/ARM-software/ML-KWS-for-MCU))


## Quy trình tiền xử lý cuối cùng

**Tập KEYWORD (`on/off/up/down`):**
  1. Load waveform, xác nhận 16kHz (Kaggle đã sẵn 16kHz, không cần resample).
  2. Trim khoảng lặng đầu/cuối — Warden ghi nhận không phải mọi file Kaggle đều đúng chính xác 1 giây, nên vẫn cần bước này dù dữ liệu đã khá sạch.
  3. (Tùy chọn) Random time-shift ±100ms trước khi trích đặc trưng — theo đúng augmentation của Warden/Sainath, tăng độ bền khi người dùng thực tế không nói đúng y hệt một mốc thời gian.
  4. Pad/cắt về đúng 1 giây (16.000 sample).
  5. (Tùy chọn) Chuẩn hóa biên độ nếu trộn nhiều nguồn có độ to nhỏ khác nhau.
  6. Trích MFCC theo thông số Hello Edge: **frame length 40ms, frame stride 20ms**, số hệ số MFCC 10–13 (paper gốc dùng 10 để tối ưu cho MCU; vì chạy trên laptop có thể dùng 13 cho chi tiết hơn).
  7. **Mở rộng so với 2 paper gốc**: thêm delta + delta-delta (đạo hàm bậc 1–2 theo thời gian) — cả Warden lẫn Zhang et al. đều không dùng bước này vì tối ưu tối đa cho MCU; ở đây có thể thêm vì chạy trên laptop không bị giới hạn tài nguyên.
  8. Lưu thành mảng `.npy`, gộp cùng nhãn.

**Tập BACKGROUND (`Background/Kaggle`, `Background/INMP441` — đã cắt sẵn 1 giây ngẫu nhiên từ file dài vài phút):**
  1. Load waveform, đã 16kHz cho cả 2 nguồn — không cần resample.
  2. **Không trim khoảng lặng** — theo đúng cách Warden xử lý: giữ nguyên trạng thái nền tự nhiên (kể cả yên tĩnh), trim sẽ phá hỏng mục đích của tập này (file im lặng có thể bị cắt gần hết, file podcast nhỏ có thể bị hiểu nhầm là "im lặng" và mất nội dung).
  3. **Không cần pad/cắt lại trong trường hợp bình thường** — file đã đúng 1 giây từ bước bốc/cắt (`Collect_KWS_Data.py`) hoặc từ script thu laptop; pipeline vẫn căn giữa phòng trường hợp file có độ dài bất thường.
  4. **Bỏ qua/giảm nhẹ chuẩn hóa biên độ theo từng file** — tránh khuếch đại nhiễu nền cực mạnh khi file gần như im lặng, làm sai lệch bản chất "yên tĩnh".
  5. Trích MFCC **với đúng thông số giống hệt tập keyword** (cùng frame length, stride, số hệ số) — bắt buộc để 2 tập cùng shape, gộp chung được vào 1 dataset.
  6. Thêm delta/delta-delta giống tập keyword.
  7. Lưu `.npy`, gộp nhãn `"background"` chung với 4 nhãn lệnh.

> Điểm mấu chốt: khác biệt giữa 2 luồng chỉ nằm ở bước xử lý waveform thô (trim/pad/normalize), phản ánh đúng bản chất khác nhau của 2 loại dữ liệu — bước trích đặc trưng MFCC cuối cùng phải dùng chung thông số để tạo ra tensor cùng shape, gộp được vào một dataset huấn luyện duy nhất.

## Bổ sung cho dữ liệu micro laptop: augmentation và oversampling tĩnh

### Vì sao phải có 30 mẫu laptop cho mỗi lớp?

Phần lớn dữ liệu đến từ Google Speech Commands và INMP441 có đặc tính microphone, mức âm lượng và môi trường khác với laptop đang chạy realtime. Vì vậy, ngoài dữ liệu nguồn, mỗi lớp `On`, `Off`, `Up`, `Down`, `Background` phải có **30 file `.wav` dài 1 giây tự thu bằng đúng micro laptop**. Các file này mang tên `laptop_*.wav` để pipeline nhận biết.

Tập âm thanh thô vẫn cân bằng 15.000 file (3.000/lớp): keyword = 2.970 Kaggle + 30 laptop; background = 1.485 Kaggle + 1.485 INMP441 + 30 laptop.

### Cách `Preprocess_KWS_Raw.py` tăng dữ liệu mà không thêm file `.wav`

`Preprocess_KWS_Raw.py` không copy hay ghi thêm audio. Khi tạo `.npy`, mỗi file `laptop_*.wav` tạo **1 feature gốc và 7 feature biến thể** (tổng 8 feature/file). Đây là oversampling ở mức feature: số lần model thấy miền âm thanh laptop tăng lên, còn dataset wav thô vẫn giữ nguyên 15.000 file.

* **Laptop keyword (`On`, `Off`, `Up`, `Down`)**: căn giữa 1 giây, time-shift tối đa ±100 ms, 85% khả năng trộn một background ngẫu nhiên với SNR 20–30 dB, sau đó peak-normalize rồi trích MFCC + delta + delta-delta.
* **Laptop background**: dịch thời gian, thay đổi gain và có thể trộn thêm background khác; không peak-normalize để giữ bản chất mức âm lượng tự nhiên của tiếng nền.
* Bộ sinh số ngẫu nhiên được seed theo đường dẫn file và chỉ số biến thể. Vì vậy các biến thể tái lập được khi chạy lại; seed này **không** quyết định train/validation/test split.

Kích thước output của pipeline hiện tại:

```text
15.000 feature gốc
+ (5 lớp × 30 file laptop × 7 feature augmentation)
= 16.050 feature = 3.210 feature/lớp

X_voice.npy : (16050, 51, 39)
Y_labels.npy: (16050,)
```

> Con số **15.840** chỉ đúng nếu augmentation 7 biến thể được áp dụng cho 4 keyword và bỏ qua 30 `Background/Laptop`. Pipeline hiện tại tăng dữ liệu laptop cho cả 5 lớp để giữ cân bằng lớp, nên con số đúng là **16.050**.

### Giới hạn cần nhớ khi đánh giá

Augmentation hiện được tạo trước rồi mới chia train/validation/test để giữ nguyên `Train_Model_KWS.py`. Vì các biến thể của một `laptop_*.wav` có thể rơi vào nhiều split, accuracy offline có thể lạc quan hơn khả năng tổng quát hóa thật. Do đó phải đánh giá thêm bằng micro laptop với câu nói, khoảng cách và tiếng nền chưa dùng để thu 30 mẫu; không chỉ dựa vào accuracy trong log train.
