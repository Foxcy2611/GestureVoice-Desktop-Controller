"""
====================
Tên script: Eval_KWS_Mic.py
Tác dụng: Chạy KWS real-time với ring buffer, gate, VAD, voting và cooldown.
====================
"""

"""
Phase 4 - GestureVoice Desktop Controller
Test model voice (DS-CNN) qua MIC LAPTOP thời gian thực (sliding window).

BẢN ĐÃ SỬA:
    1. FIX race condition: 2 dòng cập nhật write_pos/total_written trong
       Write_to_Buffer() đưa vào TRONG khối "with lock:" (bản gốc để
       ngoài lock -> đọc/ghi buffer có thể lệch nhau giữa thread callback
       và main thread).
    2. Calculate_RMS() đổi sang tính theo cửa sổ con (0.05s) lấy MAX, thay
       vì RMS trung bình cả RAW_CAPTURE_SEC (1.5s) - tránh energy gate
       loại oan các từ ngắn như "up" (chỉ chiếm ~0.2-0.3s, bị pha loãng
       nếu tính RMS trung bình cả cửa sổ dài).
    3. Giữ audio thô trong buffer, không nhân gain 30x rồi clip làm méo
       dạng sóng. Peak-normalize chỉ được thực hiện sau khi qua gate.
    4. Tự đo noise floor lúc khởi động và tạo energy gate theo từng mic,
       thay cho một ngưỡng cố định buộc người dùng phải nói to.
    5. Voting dùng confidence trung bình của chính nhãn được vote và
       xóa lịch sử khi im lặng/background để không dùng phiếu cũ.
    6. Ưu tiên endpoint WASAPI của mic trên Windows và căn giữa audio
       quanh đoạn giọng nói thực sự, thay vì cắt giữa cứng cửa sổ 1.5s.
    7. Khử noise phổ ổn định trước MFCC, giúp các từ ngắn/yếu như
       "up" không bị noise mic làm giống "off".

QUAN TRỌNG - PHẢI ĐỐI CHIẾU VỚI Preprocess_Voice_Raw.py TRƯỚC KHI CHẠY:
    Sr, Frame_Length_Ms, Frame_Stride_Ms, N_MFCC ở đây PHẢI khớp tuyệt đối
    với file dùng để train model. Lệch tham số -> model không lỗi nhưng dự
    đoán sai loạn, khó phát hiện nếu không đối chiếu kỹ.

LƯU Ý VỀ MIC LAPTOP (khác INMP441/Kaggle đã dùng lúc train):
    - Mic laptop thường thu native ở 44100Hz/48000Hz, CAO HƠN 16kHz lúc
      train -> BẮT BUỘC resample xuống 16kHz trước khi trích MFCC.
    - Mic laptop thường có AGC (Automatic Gain Control) tự động khuếch đại
      âm lượng -> biên độ "sạch/đều" hơn hẳn INMP441 thu thô và Kaggle.
    - Tiếng ồn riêng của laptop (quạt, gõ phím) khác hẳn Background/INMP441.

CÁC CẢI TIẾN ĐỘ ỔN ĐỊNH (so với bản đầu tiên):
    1. Bắt RAW_CAPTURE_SEC (1.5s) dài hơn 1s rồi mới trim + căn giữa về
       đúng 1s - tránh cắt cụt từ khóa ngắn như "up" khi nó rơi lệch nhịp
       so với thời điểm lấy mẫu cửa sổ.
    2. Energy gate: cửa sổ quá im lặng (RMS dưới ngưỡng) bị bỏ qua ngay,
       không gọi model - đỡ tốn compute và giảm nhiễu jitter lúc im lặng.
    3. Voting trên N cửa sổ predict gần nhất: chỉ chốt/hiển thị 1 nhãn khi
       đa số cửa sổ gần đây đồng thuận - giảm hẳn hiện tượng nhảy nhãn
       liên tục (jitter) giữa các lần predict sát nhau.

CÁC KÝ HIỆU
    1. Predict: Model dự đoán
    2. RMS: Root-mean-square, dùng đo năng lượng tín hiệu cho energy gate
    3. Peak Normalize: Chuẩn hóa theo biên độ cực đại [-1, 1]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import threading
import numpy as np
import sounddevice as sd
import librosa
from keras import models

# Tránh UnicodeEncodeError khi Windows/Python chọn console cp1252.
for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "P3_Train_Model", "Output_KWS", "KWS_DS_CNN_Model.keras")

NAME_CLASSES = ["on", "off", "up", "down", "background"]

Sr = 16000
Duration = 1.0
Target_Samples = int(Sr * Duration)

Frame_Length_Ms = 40
Frame_Stride_Ms = 20
N_MFCC = 13

N_FFT = int(Sr * (Frame_Length_Ms / 1000.0))
Hop_Length = int(Sr * (Frame_Stride_Ms / 1000.0))

# ==== Thông số riêng cho REAL-TIME ====
HOP_SEC = 0.1               # Cứ 0.1s predict 1 lần, không bỏ lỡ từ "up" rất ngắn
RAW_CAPTURE_SEC = 1.5       # Bắt dài hơn 1s để khi trim k bị mất keyword
CONFIDENCE_THRESHOLD = 0.7  # Confidence trung bình tối thiểu của nhãn được vote
RMS_SUB_WINDOW_SEC = 0.05   # 50ms: bắt được từ ngắn mà không bị im lặng pha loãng

# Gate được tự hiệu chỉnh trong lúc CALIBRATION_SEC đầu tiên.
# Dù calibration bị dính tiếng động, gate vẫn bị chặn trong [MIN, MAX]
# để không buộc người dùng phải ghé sát mic và nói to.
CALIBRATION_SEC = 2.0
NOISE_PERCENTILE = 30
NOISE_GATE_MULTIPLIER = 1.5
MIN_ENERGY_GATE_RMS = 0.00003
MAX_ENERGY_GATE_RMS = 0.00005

# VAD dùng ngưỡng thấp hơn energy gate để giữ lại cả phụ âm nhỏ.
VAD_NOISE_MULTIPLIER = 1.3
MIN_VAD_RMS = 0.00002
MAX_VAD_RMS = 0.00004
VAD_FRAME_SEC = 0.025
VAD_HOP_SEC = 0.010
VAD_MAX_GAP_SEC = 0.15
VAD_MARGIN_SEC = 0.12

# Spectral subtraction giảm noise ổn định của mic laptop trước khi trích MFCC.
SPECTRAL_DENOISE = True
DENOISE_N_FFT = 512
DENOISE_HOP_LENGTH = 128
DENOISE_PERCENTILE = 20
DENOISE_STRENGTH = 2.0

# None = tự chọn. Trên Windows sẽ ưu tiên endpoint WASAPI cùng tên
# với mic mặc định. Có thể gán một device index cụ thể nếu cần.
INPUT_DEVICE = None
PREFER_WASAPI = True

# False: chỉ in keyword đã xác nhận. Đổi True khi cần xem RMS,
# background, top probabilities và trạng thái voting để chẩn đoán.
VERBOSE_LOGGING = False
COMMAND_COOLDOWN_SEC = 1.5

VOTING_WINDOW = 4           # Có 4 lần predict gần nhất (trong khoảng 0.4s)
VOTING_MIN_AGREE = 2         # Đúng 2 lần thì chốt nhãn

# Peak-normalize sau khi gate đã lọc các cửa sổ quá im
# lặng/background TRƯỚC KHI tới bước normalize này, nên lo ngại "normalize
# sẽ khuếch đại background giả tạo" không còn áp dụng - mọi cửa sổ tới đây
# đều đã xác nhận có đủ năng lượng. Lúc train, tập keyword được peak-
# normalize, nên real-time cũng cần normalize để khớp phân bố biên độ,
# tránh hiện tượng "phải nói to mới nhận được".
NORMALIZE_LIVE_AUDIO = True

# ===== TRẠNG THÁI BỘ ĐỆM VÒNG =====
buffer_state = {
    "buffer": None,           # Mảng numpy chứa audio thô
    "write_pos": 0,           # Vị trí ghi hiện tại, vượt quá cuối buffer thì quay về đầu
    "total_written": 0,       # Tổng số mẫu đã được ghi tự động
    "max_samples": 0,         # Kích thước mẫu có thể chứa (Số mẫu)
    "samplerate": 0,           # Tần số lấy mẫu
    "lock": threading.Lock(),
}


def Init_Buffer(max_seconds, samplerate):
    "Khởi tạo bộ đệm vòng trước khi mở luồng thu âm"
    max_samples = int(max_seconds * samplerate)
    buffer_state["buffer"] = np.zeros(max_samples, dtype=np.float32)
    buffer_state["write_pos"] = 0
    buffer_state["total_written"] = 0
    buffer_state["max_samples"] = max_samples
    buffer_state["samplerate"] = samplerate


def Write_to_Buffer(data):
    "Ghi vào 1 đoạn audio mới vào bộ đệm"
    data = data.flatten()
    n = len(data)
    lock = buffer_state["lock"]

    # SỬA: toàn bộ phần đọc-sửa-ghi state (kể cả write_pos, total_written)
    # đưa vào TRONG khối with lock - bản gốc để 2 dòng update cuối cùng
    # NGOÀI lock, gây race condition giữa thread callback ghi và main
    # thread đọc (Read_Last_1s), có thể đọc phải buffer bị lệch/rách.
    with lock:
        buffer      = buffer_state["buffer"]
        max_samples = buffer_state["max_samples"]
        write_pos   = buffer_state["write_pos"]
        end_pos     = write_pos + n

        if end_pos <= max_samples:
            buffer[write_pos : end_pos] = data
        else:  # Nếu dữ liệu quá buff
            start_pos = max_samples - write_pos
            buffer[write_pos:] = data[:start_pos]       # Ghi phần đầu vào cuối buff
            buffer[: n - start_pos] = data[start_pos:]   # Ghi phần cuối vào đầu buff

        buffer_state["write_pos"] = end_pos % max_samples
        buffer_state["total_written"] += n


def Read_Last_1s(seconds):
    lock = buffer_state["lock"]

    with lock:
        samplerate = buffer_state["samplerate"]
        n_samples = int(seconds * samplerate)

        if buffer_state["total_written"] < n_samples:
            return None  # chưa đủ dữ liệu

        buffer = buffer_state["buffer"]
        max_samples = buffer_state["max_samples"]
        write_pos = buffer_state["write_pos"]

        start = (write_pos - n_samples) % max_samples
        if start + n_samples <= max_samples:
            return buffer[start:start + n_samples].copy()

        # Trường hợp bị wrap quanh điểm 0 của buffer
        end_part = buffer[start:].copy()
        head_part = buffer[: n_samples - len(end_part)].copy()
        return np.concatenate([end_part, head_part])


# Lịch sử dự đoán
Predict_History = []
output_state = {
    "last_command_time": 0.0,
}


def Reset_Predict_History():
    Predict_History.clear()


def Reset_Output_State():
    output_state["last_command_time"] = 0.0


def Is_In_Command_Cooldown():
    """True khi lệnh vừa được xác nhận và chưa hết thời gian chống lặp."""
    elapsed = time.monotonic() - output_state["last_command_time"]
    return elapsed < COMMAND_COOLDOWN_SEC


def Should_Print_Command():
    """Mỗi lượt phát âm chỉ in một kết quả, tránh log lặp."""
    if Is_In_Command_Cooldown():
        return False
    output_state["last_command_time"] = time.monotonic()
    return True


def Add_History_and_Vote(label, confidence):
    """
    Thêm 1 kết quả predict mới vào chuỗi nhãn liên tiếp, trả về
    (nhan_da_vote, so_phieu_dong_thuan, confidence_trung_binh).

    Nếu nhãn mới khác nhãn ngay trước đó, xóa toàn bộ phiếu cũ. Nhờ vậy
    ON không còn giữ đa số khi người dùng đã chuyển sang nói OFF (tương tự
    UP/DOWN). Chỉ chốt khi có đủ số dự đoán LIÊN TIẾP cùng nhãn.
    """
    if Predict_History and Predict_History[-1][0] != label:
        Reset_Predict_History()

    Predict_History.append((label, confidence))

    if len(Predict_History) > VOTING_WINDOW:
        Predict_History.pop(0)

    if not Predict_History:
        return None, 0, 0.0

    popular_label = Predict_History[-1][0]
    confidences = [conf for _, conf in Predict_History]
    votes = len(Predict_History)
    mean_confidence = float(np.mean(confidences))

    if votes >= VOTING_MIN_AGREE:
        return popular_label, votes, mean_confidence
    return None, votes, mean_confidence


def Load_Model_and_Label():
    model = models.load_model(MODEL_PATH)
    classes = NAME_CLASSES
    return model, classes


def Select_Input_Device():
    """Chọn endpoint input; ưu tiên WASAPI của chính mic mặc định."""
    if INPUT_DEVICE is not None:
        return int(INPUT_DEVICE), sd.query_devices(int(INPUT_DEVICE))

    default_index = int(sd.default.device[0])
    default_info = sd.query_devices(default_index)
    if not PREFER_WASAPI or os.name != "nt":
        return default_index, default_info

    # MME có thể cắt ngắn tên; phần trước dấu '(' vẫn đủ để ghép.
    name_key = str(default_info["name"]).split("(", 1)[0].strip().lower()
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) < 1:
            continue
        host_api = sd.query_hostapis(int(device["hostapi"]))["name"]
        same_microphone = name_key and name_key in str(device["name"]).lower()
        if host_api == "Windows WASAPI" and same_microphone:
            return index, device

    return default_index, default_info


def Extract_MFCC(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=Sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=Hop_Length,
    )

    delta = librosa.feature.delta(mfcc, order=1)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    feature = np.concatenate((mfcc, delta, delta_delta), axis=0)

    return feature.T


def Center_Length(y, target_len=Target_Samples):
    "Pad/Cắt về đúng độ dài, căn giữa"
    n = len(y)
    if n < target_len:
        tong_pad = target_len - n
        pad_trai = tong_pad // 2
        pad_phai = tong_pad - pad_trai
        return np.pad(y, (pad_trai, pad_phai), mode="constant")
    else:
        start = (n - target_len) // 2
        return y[start: start + target_len]


def Extract_Active_Segment(y, sr, vad_threshold):
    """Tìm cụm giọng nói chứa frame mạnh nhất và bỏ noise xung quanh."""
    if vad_threshold is None or len(y) == 0:
        return y

    frame_len = max(1, int(VAD_FRAME_SEC * sr))
    hop_len = max(1, int(VAD_HOP_SEC * sr))
    if len(y) < frame_len:
        return y

    starts = np.arange(0, len(y) - frame_len + 1, hop_len)
    frame_rms = np.asarray([
        np.sqrt(np.mean(np.square(y[start:start + frame_len]), dtype=np.float64))
        for start in starts
    ])
    peak_frame = int(np.argmax(frame_rms))
    active = frame_rms >= vad_threshold
    if not active[peak_frame]:
        return y

    # Mở rộng từ frame mạnh nhất; cho phép khoảng ngắt ngắn giữa các âm tiết.
    max_gap_frames = max(1, int(VAD_MAX_GAP_SEC / VAD_HOP_SEC))
    left = right = peak_frame
    inactive_count = 0
    for index in range(peak_frame - 1, -1, -1):
        inactive_count = 0 if active[index] else inactive_count + 1
        if inactive_count > max_gap_frames:
            break
        if active[index]:
            left = index

    inactive_count = 0
    for index in range(peak_frame + 1, len(active)):
        inactive_count = 0 if active[index] else inactive_count + 1
        if inactive_count > max_gap_frames:
            break
        if active[index]:
            right = index

    margin = int(VAD_MARGIN_SEC * sr)
    start_sample = max(0, int(starts[left]) - margin)
    end_sample = min(len(y), int(starts[right]) + frame_len + margin)
    return y[start_sample:end_sample]


def Reduce_Stationary_Noise(y):
    """Khử noise phổ bằng profile lấy từ các frame yên nhất trong đoạn."""
    if not SPECTRAL_DENOISE or len(y) < DENOISE_N_FFT:
        return y

    spectrum = librosa.stft(
        y,
        n_fft=DENOISE_N_FFT,
        hop_length=DENOISE_HOP_LENGTH,
    )
    magnitude = np.abs(spectrum)
    noise_profile = np.percentile(
        magnitude,
        DENOISE_PERCENTILE,
        axis=1,
        keepdims=True,
    )
    clean_magnitude = np.maximum(
        magnitude - DENOISE_STRENGTH * noise_profile,
        0.0,
    )
    soft_mask = clean_magnitude / (magnitude + 1e-10)
    return librosa.istft(
        spectrum * soft_mask,
        hop_length=DENOISE_HOP_LENGTH,
        length=len(y),
    ).astype(np.float32)


def Calculate_RMS_Chunks(y, sub_window_sec=RMS_SUB_WINDOW_SEC, sr=Sr):
    """Trả về RMS của từng cửa sổ con, tính theo đúng sample rate."""
    if len(y) == 0:
        return np.array([], dtype=np.float32)

    sub_len = max(1, int(sub_window_sec * sr))
    return np.asarray([
        float(np.sqrt(np.mean(np.square(y[start:start + sub_len]), dtype=np.float64)))
        for start in range(0, len(y), sub_len)
        if len(y[start:start + sub_len]) > 0
    ], dtype=np.float32)


def Calculate_RMS(y, sub_window_sec=RMS_SUB_WINDOW_SEC, sr=Sr):
    """
    Tính RMS DÙNG ĐỂ LÀM ENERGY GATE.

    SỬA so với bản gốc: bản gốc tính RMS trung bình trên TOÀN BỘ cửa sổ
    RAW_CAPTURE_SEC (1.5s). Vấn đề: từ ngắn như "up" chỉ chiếm ~0.2-0.3s
    trong đó, phần còn lại là im lặng -> RMS trung bình bị pha loãng, dễ
    tụt dưới ENERGY_GATE_RMS dù thực chất có tiếng nói rõ ràng ở đâu đó
    trong cửa sổ -> cửa sổ bị loại oan, không bao giờ tới được model.

    Cách sửa: chia cửa sổ lớn thành nhiều cửa sổ con (mặc định 0.05s),
    tính RMS từng cửa sổ con, lấy giá trị LỚN NHẤT trong số đó. Phản ánh
    đúng câu hỏi "có tồn tại 1 khoảnh khắc đủ to trong đoạn ghi hay
    không", không bị pha loãng bởi phần im lặng xung quanh từ ngắn.
    """
    rms_chunks = Calculate_RMS_Chunks(y, sub_window_sec=sub_window_sec, sr=sr)
    return float(np.max(rms_chunks)) if len(rms_chunks) else 0.0


def Calibrate_Energy_Gate(y, sr):
    """Tạo ngưỡng gate từ noise RMS khi khởi động."""
    rms_chunks = Calculate_RMS_Chunks(y, sr=sr)
    if len(rms_chunks) == 0:
        return MIN_ENERGY_GATE_RMS, 0.0

    # Dùng phần yên hơn của calibration để click/tiếng động ngắn không
    # kéo noise floor lên cao. np.clip là lớp bảo vệ thứ hai.
    noise_rms = float(np.percentile(rms_chunks, NOISE_PERCENTILE))
    gate_rms = float(np.clip(
        noise_rms * NOISE_GATE_MULTIPLIER,
        MIN_ENERGY_GATE_RMS,
        MAX_ENERGY_GATE_RMS,
    ))
    return gate_rms, noise_rms


def Processing_Number(y_goc, sr_goc, vad_threshold=None):
    """
    Xử lý 1 cửa sổ audio thô (ở sample rate gốc của mic) -> đặc trưng MFCC.
    y_goc dài RAW_CAPTURE_SEC (dư hơn 1s) để trim không cắt cụt từ khóa.
    """
    # 1. Tìm đúng cụm giọng nói trên tín hiệu thô. Việc này rất quan
    # trọng khi giọng nhỏ: trim theo top_db có thể coi cả noise là "không im".
    y_goc = Extract_Active_Segment(y_goc, sr_goc, vad_threshold)

    # 2. Resample về 16kHz (mic laptop thường 44100/48000Hz > 16kHz lúc train)
    if sr_goc != Sr:
        y = librosa.resample(y_goc, orig_sr=sr_goc, target_sr=Sr)
    else:
        y = y_goc

    # 3. Giảm noise ổn định của mic laptop. Profile noise được ước lượng
    # từ các frame có năng lượng thấp nhất trong chính đoạn đang xử lý.
    y = Reduce_Stationary_Noise(y)

    # 4. Trim khoảng lặng - vì bắt DƯ ra RAW_CAPTURE_SEC nên trim ở đây an
    # toàn hơn nhiều so với trim ngay trên đúng 1 giây (không lo cắt cụt từ)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) > 0:
        y = y_trimmed

    # 5. Căn giữa về đúng 1 giây
    y = Center_Length(y)

    # 6. Normalize - giống pipeline keyword lúc train
    if NORMALIZE_LIVE_AUDIO:
        dinh = np.max(np.abs(y))
        if dinh > 1e-6:
            y = y / dinh

    # 7. Trích đặc trưng
    return Extract_MFCC(y)


def Predict_Window(model, classes, feature):
    X = feature[np.newaxis, ...]  # Thêm chiều batch -> (1, n_frames, 39)

    expected_shape = model.input_shape
    if len(expected_shape) == 4 and X.ndim == 3:
        X = X[..., np.newaxis]  # Về chuẩn input của CNN (1, 49, 39, 1)

    probs = model.predict(X, verbose=0)[0]
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    label = str(classes[class_id])

    return label, confidence, probs


def Format_Top_Probabilities(classes, probs, top_n=3):
    top_ids = np.argsort(probs)[::-1][:top_n]
    return ", ".join(f"{classes[index]}={probs[index]*100:.1f}%" for index in top_ids)


def Record_Callbacks(indata, frames, time_info, status):
    if status:
        print(f"[WAR SoundDevice] {status}")

    # Giữ nguyên tín hiệu thô. Nhân gain lớn rồi clip tại đây sẽ làm
    # méo phụ âm, trong khi Processing_Number đã peak-normalize sau gate.
    Write_to_Buffer(indata[:, 0])


def main():
    print("Đang tải model và bảng nhãn...")
    model, classes = Load_Model_and_Label()
    print(f"  + Model: {MODEL_PATH}")
    print(f"  + Các lớp: {list(classes)}")
    print(f"  + Model input shape: {model.input_shape}")

    # Chọn endpoint thu âm. WASAPI thường ít lớp chuyển đổi hơn MME.
    input_device, thong_tin_mic = Select_Input_Device()
    sr_goc = int(thong_tin_mic["default_samplerate"])
    host_api = sd.query_hostapis(int(thong_tin_mic["hostapi"]))["name"]
    print(f"\nMic đang dùng: [{input_device}] {thong_tin_mic['name']} ({host_api})")
    print(f"Sample rate gốc của mic: {sr_goc}Hz -> sẽ resample xuống {Sr}Hz")
    if sr_goc == Sr:
        print("  (Trùng hợp mic đã sẵn 16kHz - bước resample sẽ không làm gì cả)")

    Init_Buffer(
        max_seconds=max(CALIBRATION_SEC + 0.5, RAW_CAPTURE_SEC + 0.5),
        samplerate=sr_goc,
    )

    with sd.InputStream(
        device=input_device,
        samplerate=sr_goc,
        channels=1,
        dtype="float32",
        callback=Record_Callbacks,
    ):
        print(f"\nĐang đo noise nền trong {CALIBRATION_SEC:.1f}s - vui lòng giữ im lặng...")
        time.sleep(CALIBRATION_SEC)
        calibration_audio = Read_Last_1s(CALIBRATION_SEC)
        while calibration_audio is None:
            # Callback có thể chưa giao đủ chính xác 2 giây mẫu khi sleep kết thúc.
            time.sleep(0.05)
            calibration_audio = Read_Last_1s(CALIBRATION_SEC)
        energy_gate_rms, noise_rms = Calibrate_Energy_Gate(calibration_audio, sr_goc)
        vad_threshold = float(np.clip(
            noise_rms * VAD_NOISE_MULTIPLIER,
            MIN_VAD_RMS,
            MAX_VAD_RMS,
        ))
        print(f"  + Noise RMS: {noise_rms:.6f}")
        print(f"  + Energy gate tự động: {energy_gate_rms:.6f} "
              f"(P{NOISE_PERCENTILE} x{NOISE_GATE_MULTIPLIER:.1f}, "
              f"giới hạn={MIN_ENERGY_GATE_RMS:.6f}..{MAX_ENERGY_GATE_RMS:.6f})\n")
        print(f"  + VAD threshold: {vad_threshold:.6f}\n")
        print(f"Đang lắng nghe (bắt {RAW_CAPTURE_SEC}s, predict mỗi {HOP_SEC}s, "
              f"voting {VOTING_MIN_AGREE}/{VOTING_WINDOW}).")
        print(f"Chế độ log: {'chi tiết' if VERBOSE_LOGGING else 'gọn - chỉ hiện keyword'}.")
        print("Nói rõ các từ: on / off / up / down. Ctrl+C để thoát.\n")

        try:
            while True:
                time.sleep(HOP_SEC)

                y_goc = Read_Last_1s(RAW_CAPTURE_SEC)
                if y_goc is None:
                    continue  # chưa đủ dữ liệu, chờ thêm

                # Energy gate: cửa sổ quá im lặng thì bỏ qua ngay, không predict
                rms = Calculate_RMS(y_goc, sr=sr_goc)

                if VERBOSE_LOGGING:
                    trang_thai = "BI CHAN" if rms < energy_gate_rms else "QUA GATE"
                    print(f"[DEBUG RMS  ] rms={rms:.6f}  nguong={energy_gate_rms:.6f}  ({trang_thai})")

                if rms < energy_gate_rms:
                    Reset_Predict_History()
                    Reset_Output_State()
                    if VERBOSE_LOGGING:
                        print(f"[im lặng    ] rms={rms:.6f} "
                              f"(dưới ngưỡng {energy_gate_rms:.6f}, bỏ qua)")
                    continue

                # Trong cooldown không chạy model và không tích phiếu mới.
                # Nếu vẫn tích phiếu, cửa sổ 1.5 giây còn chứa keyword trước
                # sẽ làm lịch sử lại đầy nhãn cũ và kéo sai lệnh kế tiếp.
                if Is_In_Command_Cooldown():
                    Reset_Predict_History()
                    if VERBOSE_LOGGING:
                        print("[cooldown    ] bỏ qua cửa sổ còn chứa lệnh trước")
                    continue

                t0 = time.time()
                dac_trung = Processing_Number(y_goc, sr_goc, vad_threshold=vad_threshold)
                label_tho, confidence, probs = Predict_Window(model, classes, dac_trung)
                do_tre_ms = (time.time() - t0) * 1000
                top_probs = Format_Top_Probabilities(classes, probs)

                if label_tho == "background":
                    Reset_Predict_History()
                    if VERBOSE_LOGGING:
                        print(f"[background  ] conf={confidence*100:5.1f}%  "
                              f"top=[{top_probs}]  (xử lý: {do_tre_ms:.0f}ms)")
                    continue

                nhan_da_vote, so_phieu, vote_confidence = Add_History_and_Vote(
                    label_tho, confidence
                )

                if nhan_da_vote is None:
                    if VERBOSE_LOGGING:
                        print(f"[đang chờ   ] gợi ý={label_tho} conf={confidence*100:5.1f}%  "
                              f"top=[{top_probs}]  "
                              f"(chưa đủ {VOTING_MIN_AGREE}/{VOTING_WINDOW} đồng thuận, "
                              f"xử lý: {do_tre_ms:.0f}ms)")
                elif vote_confidence >= CONFIDENCE_THRESHOLD:
                    if Should_Print_Command():
                        print(f"[{nhan_da_vote:12s}] conf={vote_confidence*100:5.1f}%  "
                              f"phieu={so_phieu}/{VOTING_WINDOW}")
                        Reset_Predict_History()
                else:
                    if VERBOSE_LOGGING:
                        print(f"[?           ] conf={vote_confidence*100:5.1f}%  "
                              f"(gần nhất: {nhan_da_vote}, xử lý: {do_tre_ms:.0f}ms)")

        except KeyboardInterrupt:
            print("\nĐã dừng.")


if __name__ == "__main__":
    main()
