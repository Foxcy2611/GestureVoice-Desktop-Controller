# Hướng dẫn cơ bản sử dụng thư viện Librosa (Python)

`librosa` là thư viện xử lý tín hiệu âm thanh cho Python. Trong project này, thư viện được dùng để chuẩn hóa audio về 16 kHz, cắt khoảng lặng, chuẩn hóa biên độ và trích xuất MFCC cho Keyword Spotting (KWS).

Tài liệu tham khảo chính thức: [Librosa Documentation](https://librosa.org/doc/latest/index.html).

## 1. Import thư viện và đọc file audio

```python
import librosa
import numpy as np

# Đọc WAV, chuyển thành mono và resample trực tiếp về 16 kHz.
y, sr = librosa.load(
    "audio.wav",
    sr=16000,
    mono=True,
    dtype=np.float32,
)
```

### Các tham số quan trọng của `librosa.load`

* `path`: đường dẫn tới file audio.
* `sr=16000`: sample rate đầu ra. Nếu file gốc là 48 kHz, Librosa tự resample về 16 kHz.
* `mono=True`: nếu file có nhiều kênh, Librosa trộn về một kênh mono.
* `dtype=np.float32`: dùng kiểu số thực 32-bit, phù hợp cho NumPy và TensorFlow/Keras.
* `sr=None`: giữ nguyên sample rate gốc. Chỉ dùng khi cần tự resample ở bước sau.

Với KWS, nên thống nhất toàn bộ audio ở `16_000 Hz`; nếu không, cùng một frame 40 ms của hai file sẽ có số sample khác nhau.

## 2. Resample audio đã có sẵn trong bộ nhớ

Khi audio đến từ microphone hoặc `sounddevice`, dữ liệu đã là mảng NumPy. Khi đó dùng `librosa.resample` thay vì `librosa.load`:

```python
# y_raw được thu từ mic ở 48 kHz.
y_16k = librosa.resample(
    y_raw,
    orig_sr=48000,
    target_sr=16000,
)
```

* `orig_sr`: sample rate thực tế của microphone.
* `target_sr`: sample rate model yêu cầu, trong project là 16 kHz.

Không cần gọi `librosa.resample` sau `librosa.load(..., sr=16000)`, vì file đã được resample ở lúc load.

## 3. Cắt khoảng lặng đầu và cuối bằng `effects.trim`

Keyword thường có khoảng lặng trước/sau khi nói. Ta trim để từ khóa được tập trung hơn trong cửa sổ 1 giây:

```python
y_trimmed, index = librosa.effects.trim(y, top_db=30)

print(index)  # [vị_trí_bắt_đầu, vị_trí_kết_thúc] trong waveform gốc
```

### `top_db=30` có ý nghĩa gì?

* Librosa tính năng lượng theo frame audio.
* Frame có năng lượng thấp hơn khoảng 30 dB so với frame mạnh nhất được coi là khoảng lặng và bị cắt.
* Giá trị thấp hơn như `20` cắt mạnh hơn; giá trị cao hơn như `40` giữ lại nhiều noise hơn.

Trong KWS, chỉ trim tập keyword `On/Off/Up/Down`. Không trim `Background`, vì sự yên lặng và noise chính là nội dung mà lớp background cần học.

## 4. Pad hoặc cắt audio về đúng 1 giây

Model KWS chỉ nhận một waveform dài đúng 1 giây:

```python
SR = 16000
TARGET_SAMPLES = SR * 1  # 16_000 sample = 1 giây

def Center_One_Second(y):
    if len(y) > TARGET_SAMPLES:
        start = (len(y) - TARGET_SAMPLES) // 2
        return y[start:start + TARGET_SAMPLES]

    if len(y) < TARGET_SAMPLES:
        total_pad = TARGET_SAMPLES - len(y)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return np.pad(y, (left_pad, right_pad), mode="constant")

    return y
```

* Audio dài hơn 1 giây: cắt phần giữa.
* Audio ngắn hơn 1 giây: thêm số 0 ở hai đầu để căn giữa từ khóa.
* Audio đúng 1 giây: giữ nguyên.

## 5. Chuẩn hóa biên độ với `util.normalize`

Các file thu từ mic laptop có thể nhỏ hơn nhiều so với file Kaggle. Với keyword, ta có thể chuẩn hóa peak trước khi trích MFCC:

```python
y_normalized = librosa.util.normalize(y)
```

Với waveform một chiều, giá trị biên độ tuyệt đối lớn nhất sẽ gần bằng `1.0`.

Không nên peak-normalize từng file `Background`: một đoạn gần như im lặng sẽ bị khuếch đại thành noise lớn và làm sai ý nghĩa của lớp background.

## 6. Trích xuất MFCC cho Keyword Spotting

MFCC (Mel-Frequency Cepstral Coefficients) biến waveform thành ma trận đặc trưng theo thời gian và tần số:

```python
SR = 16000
FRAME_LENGTH_MS = 40
FRAME_STRIDE_MS = 20
N_MFCC = 13

N_FFT = int(SR * FRAME_LENGTH_MS / 1000.0)       # 640 sample
HOP_LENGTH = int(SR * FRAME_STRIDE_MS / 1000.0)  # 320 sample

mfcc = librosa.feature.mfcc(
    y=y_normalized,
    sr=SR,
    n_mfcc=N_MFCC,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)

print(mfcc.shape)  # (13, 51) với audio 1 giây ở 16 kHz
```

* `n_fft=640`: cửa sổ 40 ms.
* `hop_length=320`: bước trượt 20 ms.
* `n_mfcc=13`: 13 hệ số MFCC mỗi frame.
* Kết quả có 51 frame do Librosa mặc định pad ở hai đầu khi tính phổ.

## 7. Delta và Delta-Delta

MFCC mô tả phổ tại từng thời điểm. Delta mô tả tốc độ thay đổi của MFCC, còn delta-delta mô tả mức thay đổi của delta:

```python
delta = librosa.feature.delta(mfcc, order=1)
delta_delta = librosa.feature.delta(mfcc, order=2)

# Ghép 13 MFCC + 13 delta + 13 delta-delta = 39 đặc trưng/frame.
feature = np.concatenate((mfcc, delta, delta_delta), axis=0).T

print(feature.shape)  # (51, 39)
```

Ma trận `(51, 39)` là đầu vào của DS-CNN trong `Train_Model_KWS.py`.

## 8. STFT và ISTFT

STFT (Short-Time Fourier Transform) chuyển audio sang miền thời gian–tần số phức. Trong project, nó được dùng cho spectral denoise ở realtime:

```python
spec = librosa.stft(
    y,
    n_fft=512,
    hop_length=128,
)

magnitude = np.abs(spec)
phase = np.angle(spec)

# Sau khi xử lý magnitude, ghép lại waveform.
processed_spec = magnitude * np.exp(1j * phase)
y_restored = librosa.istft(
    processed_spec,
    hop_length=128,
    length=len(y),
)
```

* `librosa.stft`: waveform -> phổ phức.
* `np.abs(spec)`: biên độ của từng dải tần.
* `np.angle(spec)`: pha của từng dải tần.
* `librosa.istft`: phổ phức -> waveform.

## 9. Pipeline Librosa áp dụng cho KWS

```text
WAV / microphone
        ↓
Load hoặc resample về 16 kHz, mono
        ↓
Keyword: trim -> căn 1 giây -> time-shift -> normalize
Background: giữ biên độ tự nhiên
        ↓
MFCC 13 hệ số + Delta + Delta-Delta
        ↓
Feature (51, 39) -> X_voice.npy
```

Điểm quan trọng nhất: train và realtime phải dùng cùng sample rate, thông số MFCC và cách chuẩn hóa keyword. Nếu các bước này khác nhau, model dễ giảm độ chính xác dù dữ liệu train tốt.
