"""
====================
Tên script: Preprocess_KWS_Raw.py
Tác dụng: Tiền xử lý WAV, augmentation và trích MFCC/delta/delta-delta cho KWS.
====================
"""

"""
Tiền xử lý dữ liệu Keyword Spotting (KWS).

Mỗi WAV gốc sinh ra một ma trận (51, 39):
    13 MFCC + 13 Delta + 13 Delta-Delta.

Pipeline gốc bám theo tài liệu Preprocess_Audio.md:
    keyword: load -> trim -> canvas 1 giây + time-shift -> normalize -> MFCC;
    background: load -> giữ nguyên biên độ -> MFCC.

Riêng file laptop_*.wav được oversampling tĩnh vì Train_Model_KWS.py
đọc trực tiếp NPY và không augmentation động theo từng epoch:
    1 bản gốc + 7 bản augmentation = 8 feature/file.

Keyword laptop được augmentation bằng:
    - time-shift có zero-padding;
    - trộn noise từ Background/Kaggle, INMP441 và Laptop;
    - peak-normalize trước khi trích MFCC.

Output chỉ gồm hai file, tương thích trực tiếp Train_Model_KWS.py:
    X_voice.npy
    Y_labels.npy
"""

import glob
import os
import sys
import zlib

import librosa
import numpy as np


for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# 1. CẤU HÌNH
# =============================================================================

SR = 16000
DURATION_SEC = 1.0
TARGET_SAMPLES = int(SR * DURATION_SEC)

FRAME_LENGTH_MS = 40
FRAME_STRIDE_MS = 20
N_MFCC = 13
N_FFT = int(SR * FRAME_LENGTH_MS / 1000.0)       # 640 samples
HOP_LENGTH = int(SR * FRAME_STRIDE_MS / 1000.0)  # 320 samples

LABEL_MAP = {
    "on": 0,
    "off": 1,
    "up": 2,
    "down": 3,
    "background": 4,
}

# Dataset gốc sau khi thu xong phải có 3.000 WAV/lớp:
# 2.970 file nguồn + 30 file laptop.
EXPECTED_BASE_FILES_PER_CLASS = 3000
EXPECTED_LAPTOP_FILES_PER_CLASS = 30 
EXPECTED_BG_KAGGLE = 1485
EXPECTED_BG_INMP441 = 1485

# 1 bản gốc + 7 augmentation = 8 feature cho mỗi WAV laptop.
LAPTOP_TOTAL_VERSIONS = 8
RANDOM_SEED = 42

# Theo tài liệu: time-shift tối đa ±100 ms. Noise ở SNR 20-30 dB
# tương đương khoảng 3-10% RMS của keyword, không lấn át tiếng nói.
MAX_SHIFT_MS = 100
AUGMENT_SNR_DB_RANGE = (20.0, 30.0)
AUGMENT_NOISE_PROBABILITY = 0.85

MAX_SHIFT_SAMPLES = int(SR * MAX_SHIFT_MS / 1000.0)


# =============================================================================
# 2. HÀM AUDIO VÀ TRÍCH ĐẶC TRƯNG
# =============================================================================

# Tạo ra bộ số ngẫu nhiên cho từng file và từng bản augmentation
# Giúp các file được lựa chọn cách mà nó được augmentation
def Make_Rng(file_path, variant_index=0):
    """Random theo file/variant, giúp chạy lại cho kết quả giống nhau."""
    path_hash = zlib.crc32(os.path.abspath(file_path).encode("utf-8"))
    seed = (RANDOM_SEED + path_hash + variant_index * 10007) % (2**32)
    return np.random.default_rng(seed)


def Is_Laptop_File(file_path):
    return os.path.basename(file_path).lower().startswith("laptop_")


def Center_One_Second(y):
    """Pad zero hai đầu hoặc cắt giữa audio về đúng một giây."""
    if len(y) > TARGET_SAMPLES:
        start = (len(y) - TARGET_SAMPLES) // 2      # Chia nửa đoạn dư
        return y[start : (start + TARGET_SAMPLES)]  # Cắt lấy từ nửa đoạn dư đến gần cuối đoạn 

    if len(y) < TARGET_SAMPLES:
        total_pad = TARGET_SAMPLES - len(y)
        left_pad = total_pad // 2
        return np.pad(y, (left_pad, total_pad - left_pad), mode="constant")

    return y


def Load_Audio(file_path):
    """Đọc WAV, chuyển mono và resample trực tiếp về chuẩn 16 kHz."""
    y, _ = librosa.load(
        file_path,
        sr=SR,
        mono=True,
        dtype=np.float32,
    )
    return y


def Time_Shift_Zero_Pad(y, shift_samples):
    """Dịch audio, zero-pad và giữ nguyên độ dài hiện tại của waveform."""
    original_length = len(y)
    if original_length == 0:
        return y

    # Tính số mẫu dịch thực tế, đảm bảo k vượt quá độ dài audio
    amount = min(abs(int(shift_samples)), original_length)

    if shift_samples > 0:
        # Thêm amount số 0 ở đầu
        # Cắt về đúng độ dài gốc
        # Kết quả waveform bị trễ
        return np.pad(y, (amount, 0), mode="constant")[:original_length]

    if shift_samples < 0:
        # Bỏ đi amount mẫu đầu 
        # Thêm amount số 0 ở cuối
        # Kết quả waveform bị sớm
        return np.pad(y[amount:], (0, amount), mode="constant")[:original_length]

    return y


def Extract_Features(y):
    """Trích MFCC + Delta + Delta-Delta, trả về shape (51, 39)."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    delta = librosa.feature.delta(mfcc, order=1)

    delta_delta = librosa.feature.delta(mfcc, order=2)

    feature = np.concatenate((mfcc, delta, delta_delta), axis=0)
    return feature.T


def Load_Keyword(file_path):
    """Load 16 kHz, mono và trim."""
    y = Load_Audio(file_path)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    return y_trimmed if len(y_trimmed) else y


def Load_Background_One_Second(file_path, rng):
    """Load background; nếu file dài thì cắt ngẫu nhiên một đoạn 1 giây."""
    y = Load_Audio(file_path)

    if len(y) > TARGET_SAMPLES:
        start = int(rng.integers(0, len(y) - TARGET_SAMPLES + 1))
        return y[start:start + TARGET_SAMPLES]

    return Center_One_Second(y)


def Mix_Noise_At_SNR(keyword, background, snr_db):
    """Trộn background theo SNR mong muốn, không làm thay đổi nhãn."""
    keyword = Center_One_Second(keyword).astype(np.float32)
    background = Center_One_Second(background).astype(np.float32)

    keyword_rms = float(np.sqrt(np.mean(np.square(keyword), dtype=np.float64)))
    noise_rms = float(np.sqrt(np.mean(np.square(background), dtype=np.float64)))
    if keyword_rms <= 1e-10 or noise_rms <= 1e-10:
        return keyword

    target_noise_rms = keyword_rms / (10.0 ** (snr_db / 20.0))
    return (keyword + background * (target_noise_rms / noise_rms)).astype(np.float32)


# =============================================================================
# 3. PIPELINE CHO FILE GỐC VÀ FILE AUGMENTATION
# =============================================================================

def Preprocess_Keyword(file_path):
    """Một file keyword gốc -> một feature (51, 39)."""
    rng = Make_Rng(file_path, variant_index=0)
    y = Load_Keyword(file_path)

    # Đặt keyword đã trim lên canvas 1 giây rồi mới shift. Cách này hiện thực
    # an toàn hai bước time-shift + pad/cắt, không cắt phụ âm ngay trên đoạn trim.
    y = Center_One_Second(y)
    shift = int(rng.integers(-MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES + 1))
    y = Time_Shift_Zero_Pad(y, shift)

    y = librosa.util.normalize(y)
    return Extract_Features(y)


def Preprocess_Background(file_path):
    """Một file background gốc -> một feature; không peak-normalize noise."""
    rng = Make_Rng(file_path, variant_index=0)

    y = Load_Background_One_Second(file_path, rng)
    
    return Extract_Features(y)


def Augment_Laptop_Keyword(file_path, background_files, variant_index):
    """Tạo một biến thể mới từ keyword laptop."""
    rng = Make_Rng(file_path, variant_index)
    y = Load_Keyword(file_path)

    y = Center_One_Second(y)
    shift = int(rng.integers(-MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES + 1))
    y = Time_Shift_Zero_Pad(y, shift)

    # Noise được lấy từ cả Kaggle, INMP441 và laptop.
    if background_files and rng.random() < AUGMENT_NOISE_PROBABILITY:
        bg_path = background_files[int(rng.integers(0, len(background_files)))]
        background = Load_Background_One_Second(bg_path, rng)
        snr_db = float(rng.uniform(*AUGMENT_SNR_DB_RANGE))
        y = Mix_Noise_At_SNR(y, background, snr_db)

    y = librosa.util.normalize(y)
    return Extract_Features(y)


def Augment_Laptop_Background(file_path, background_files, variant_index):
    """Tạo biến thể background bằng shift, gain và trộn nhẹ noise khác."""
    rng = Make_Rng(file_path, variant_index)
    y = Load_Background_One_Second(file_path, rng)

    shift = int(rng.integers(-MAX_SHIFT_SAMPLES, MAX_SHIFT_SAMPLES + 1))
    y = Time_Shift_Zero_Pad(y, shift)
    y = y * float(rng.uniform(0.5, 1.5))

    if background_files:
        other_path = background_files[int(rng.integers(0, len(background_files)))]
        other = Load_Background_One_Second(other_path, rng)
        y = y + other * float(rng.uniform(0.05, 0.20))

    # Chỉ chống clipping, không peak-normalize background.
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak

    return Extract_Features(y.astype(np.float32))


# =============================================================================
# 4. KIỂM TRA VÀ XÂY DỰNG X_voice.npy / Y_labels.npy
# =============================================================================

def Count_Wav(folder):
    return len(glob.glob(os.path.join(folder, "*.wav")))


def Validate_Dataset(keyword_dirs, background_dirs):
    """Dừng sớm nếu dataset chưa đủ/cân bằng trước khi tạo file NPY lớn."""
    errors = []

    for label, folder in keyword_dirs.items():
        total = Count_Wav(folder)
        laptop = len(glob.glob(os.path.join(folder, "laptop_*.wav")))
        if total != EXPECTED_BASE_FILES_PER_CLASS or laptop != EXPECTED_LAPTOP_FILES_PER_CLASS:
            errors.append(
                f"{label}: tổng={total}/{EXPECTED_BASE_FILES_PER_CLASS}, "
                f"laptop={laptop}/{EXPECTED_LAPTOP_FILES_PER_CLASS}"
            )

    bg_counts = {os.path.basename(folder).lower(): Count_Wav(folder) for folder in background_dirs}
    bg_total = sum(bg_counts.values())
    if bg_total != EXPECTED_BASE_FILES_PER_CLASS:
        errors.append(f"background: tổng={bg_total}/{EXPECTED_BASE_FILES_PER_CLASS}")
    if bg_counts.get("kaggle", 0) != EXPECTED_BG_KAGGLE:
        errors.append(f"background/kaggle={bg_counts.get('kaggle', 0)}/{EXPECTED_BG_KAGGLE}")
    if bg_counts.get("inmp441", 0) != EXPECTED_BG_INMP441:
        errors.append(f"background/inmp441={bg_counts.get('inmp441', 0)}/{EXPECTED_BG_INMP441}")
    if bg_counts.get("laptop", 0) != EXPECTED_LAPTOP_FILES_PER_CLASS:
        errors.append(
            f"background/laptop={bg_counts.get('laptop', 0)}/"
            f"{EXPECTED_LAPTOP_FILES_PER_CLASS}"
        )

    if errors:
        raise RuntimeError("Dataset chưa đúng cấu hình:\n  - " + "\n  - ".join(errors))


def Append_Feature(X, Y, feature, label_id, file_path):
    """Kiểm tra shape trước khi thêm, tránh tạo NPY lỗi âm thầm."""
    if feature.shape != (51, 39):
        raise ValueError(f"Feature {feature.shape}, cần (51, 39): {file_path}")
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"Feature chứa NaN/Inf: {file_path}")

    X.append(feature.astype(np.float32))
    Y.append(label_id)


def Build_Numpy(keyword_dirs, background_dirs, output_x, output_y):
    Validate_Dataset(keyword_dirs, background_dirs)

    # Pool noise dùng khi augmentation keyword và background laptop.
    all_background_files = sorted([
        file_path
        for folder in background_dirs
        for file_path in glob.glob(os.path.join(folder, "*.wav"))
    ])

    X = []
    Y = []

    # ---------- On / Off / Up / Down ----------
    for label, folder in keyword_dirs.items():
        label_id = LABEL_MAP[label]
        wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        print(f"[{label.upper()}] {len(wav_files)} WAV gốc")

        for file_path in wav_files:
            Append_Feature(X, Y, Preprocess_Keyword(file_path), label_id, file_path)

            # Chỉ oversampling file tự thu bằng laptop.
            if Is_Laptop_File(file_path):
                for variant_index in range(1, LAPTOP_TOTAL_VERSIONS):
                    feature = Augment_Laptop_Keyword(
                        file_path,
                        all_background_files,
                        variant_index,
                    )
                    Append_Feature(X, Y, feature, label_id, file_path)

    # ---------- Background ----------
    bg_label_id = LABEL_MAP["background"]
    for folder in background_dirs:
        wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        print(f"[BACKGROUND/{os.path.basename(folder)}] {len(wav_files)} WAV gốc")

        for file_path in wav_files:
            Append_Feature(X, Y, Preprocess_Background(file_path), bg_label_id, file_path)

            if Is_Laptop_File(file_path):
                for variant_index in range(1, LAPTOP_TOTAL_VERSIONS):
                    feature = Augment_Laptop_Background(
                        file_path,
                        all_background_files,
                        variant_index,
                    )
                    Append_Feature(X, Y, feature, bg_label_id, file_path)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int64)

    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Phân bố lớp: {np.bincount(Y, minlength=len(LABEL_MAP))}")

    np.save(output_x, X)
    np.save(output_y, Y)
    print(f"Đã lưu: {output_x}")
    print(f"Đã lưu: {output_y}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(
        project_root,
        "1_Data_Collection", "Dataset", "Voice_Data_Raw",
    )

    keyword_dirs = {
        "on": os.path.join(dataset_dir, "On"),
        "off": os.path.join(dataset_dir, "Off"),
        "up": os.path.join(dataset_dir, "Up"),
        "down": os.path.join(dataset_dir, "Down"),
    }
    background_dirs = [
        os.path.join(dataset_dir, "Background", "Kaggle"),
        os.path.join(dataset_dir, "Background", "INMP441"),
        os.path.join(dataset_dir, "Background", "Laptop"),
    ]

    output_dir = os.path.join(project_root, "2_Data_Preprocessing", "Output_npy")
    os.makedirs(output_dir, exist_ok=True)

    Build_Numpy(
        keyword_dirs=keyword_dirs,
        background_dirs=background_dirs,
        output_x=os.path.join(output_dir, "X_voice.npy"),
        output_y=os.path.join(output_dir, "Y_labels.npy"),
    )


if __name__ == "__main__":
    main()

"""
X_voice.npy: (16050, 51, 39)
Y_labels.npy: (16050,)
Phân bố lớp: [3210 3210 3210 3210 3210]
"""
