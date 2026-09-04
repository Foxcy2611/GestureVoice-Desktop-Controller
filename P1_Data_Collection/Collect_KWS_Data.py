"""
====================
Tên script: Collect_KWS_Data.py
Tác dụng: Chọn và cắt dữ liệu âm thanh thô cho bốn keyword cùng lớp background.
====================
"""

"""
GestureVoice Desktop Controller - Phase 1 (Voice)
Script bốc/cắt dữ liệu thô cho Voice_Data_Raw. CHƯA tiền xử lý (resample/trim/MFCC).

Việc script làm:
    1. on/off/up/down: bốc 2.970 file Google Speech Commands cho mỗi lớp.
    2. Background/Kaggle: cắt 1.485 đoạn 1 giây từ _background_noise_.
    3. Background/INMP441: cắt 1.485 đoạn 1 giây từ tuthu.

Sau đó Record_KWS_Laptop.py bổ sung 30 mẫu/lớp để mỗi lớp
có tổng cộng 3.000 mẫu. Script này không tự xóa file cũ.

"""

import os
import random
import shutil
import glob
import soundfile as sf

# ================== ĐƯỜNG DẪN FILE ==================

# Dùng đường dẫn tương đối để repo có thể di chuyển sang máy/thư mục khác.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục Dataset gốc, chứa sẵn: on/ off/ up/ down/ tuthu/ _background_noise_/
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset_Kaggle")

# Thư mục đích
OUT_DIR = os.path.join(SCRIPT_DIR, "Dataset", "Voice_Data_Raw")

# 4 lớp lệnh: tên folder trong Dataset -> tên folder đích
COMMANDS = {
    "on": "On",
    "off": "Off",
    "up": "Up",
    "down": "Down",
}
N_COMMAND = 2970  # + 30 mẫu laptop = 3.000 file cho MỖI lệnh

BG_KAGGLE_DIR = os.path.join(DATASET_DIR, "_background_noise_")
BG_OWN_DIR = os.path.join(DATASET_DIR, "tuthu")

# Background tách riêng 2 folder vì tiền xử lý sau này khác nhau chút giữa
# nguồn Kaggle và nguồn tự thu INMP441
BG_KAGGLE_OUT_DIR = os.path.join(OUT_DIR, "Background", "Kaggle")
BG_OWN_OUT_DIR = os.path.join(OUT_DIR, "Background", "INMP441")

N_BG_TOTAL = 2970
N_BG_KAGGLE = N_BG_TOTAL // 2       # 1.485 mẫu từ _background_noise_
N_BG_OWN = N_BG_TOTAL - N_BG_KAGGLE # 1.485 mẫu từ tuthu

SEGMENT_SEC = 1.0
SEED = 42

# =========================================================================

random.seed(SEED)


# ---------- Bước 1: bốc ngẫu nhiên file lệnh (on/off/up/down) ----------

def copy_random_commands():
    for src_name, dst_name in COMMANDS.items():
        src_dir = os.path.join(DATASET_DIR, src_name)
        dst_dir = os.path.join(OUT_DIR, dst_name)
        os.makedirs(dst_dir, exist_ok=True)

        all_files = glob.glob(os.path.join(src_dir, "*.wav"))
        n = min(N_COMMAND, len(all_files))
        if n < N_COMMAND:
            print(f"[WARN] '{src_name}' chỉ có {len(all_files)} file, lấy hết {n}")

        chosen = random.sample(all_files, n)
        for f in chosen:
            shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

        print(f"[OK] {src_name} -> {dst_name}: copy xong {n} file")


# ---------- Bước 2 & 3: cắt đoạn 1 giây từ file dài ----------

# Khoảng cách tối thiểu giữa 2 điểm bắt đầu trong CÙNG 1 file, tính theo % độ dài
# đoạn cắt. 1.0 = 2 đoạn không được chồng lấn chút nào. 0.5 = cho phép chồng
# lấn tối đa 50%. Đặt nhỏ hơn nếu file ngắn/ít file mà cần nhiều mẫu.
MIN_GAP_RATIO = 1.0
MAX_RETRY = 200  # số lần thử lại tối đa cho 1 đoạn trước khi bỏ qua giới hạn khoảng cách


def cut_random_segments(src_dir, out_dir, n_needed, prefix):
    """Cắt ngẫu nhiên n_needed đoạn 1 giây từ các file .wav trong src_dir, lưu ra out_dir.
    Có kiểm tra khoảng cách tối thiểu giữa các lần cắt trong cùng 1 file để tránh
    2 đoạn chồng lấn gần như trùng nhau."""
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(src_dir, "*.wav"))
    if not files:
        raise RuntimeError(f"Không tìm thấy file .wav nào trong {src_dir}")
    print(f"[INFO] {src_dir}: tìm thấy {len(files)} file")

    used_starts = {f: [] for f in files}  # lưu các điểm bắt đầu đã dùng, theo từng file
    saved = 0
    gap_ratio = MIN_GAP_RATIO
    fail_streak = 0  # đếm số lần liên tiếp không tìm được chỗ trống

    while saved < n_needed:
        f = random.choice(files)
        data, sr = sf.read(f)
        if data.ndim > 1:
            data = data[:, 0]  # nếu stereo thì lấy 1 kênh

        seg_len = int(SEGMENT_SEC * sr)
        if len(data) < seg_len:
            continue  # file quá ngắn, bỏ qua

        min_gap = int(seg_len * gap_ratio)

        # Thử tìm 1 điểm bắt đầu cách xa các điểm đã dùng trong file này
        start = None
        for _ in range(MAX_RETRY):
            candidate = random.randint(0, len(data) - seg_len)
            if all(abs(candidate - u) >= min_gap for u in used_starts[f]):
                start = candidate
                break

        if start is None:
            fail_streak += 1
            # Nếu thử nhiều lần liên tiếp mà mọi file đều kín chỗ -> nới lỏng
            # khoảng cách tối thiểu (chấp nhận chồng lấn nhiều hơn) để không treo
            if fail_streak >= len(files) * 3:
                gap_ratio = max(gap_ratio * 0.5, 0.0)
                fail_streak = 0
                print(f"[INFO] Các file gần kín chỗ, nới khoảng cách tối thiểu "
                      f"xuống {gap_ratio:.2f}x độ dài đoạn để đủ {n_needed} mẫu")
            continue

        fail_streak = 0
        used_starts[f].append(start)
        segment = data[start:start + seg_len]

        out_name = f"{prefix}_{saved:05d}.wav"
        sf.write(os.path.join(out_dir, out_name), segment, sr)
        saved += 1

    print(f"[OK] Đã cắt {saved} đoạn vào {out_dir} (prefix={prefix})")


def count_wav_files(folder, recursive=False):
    pattern = os.path.join(folder, "**", "*.wav") if recursive else os.path.join(folder, "*.wav")
    return len(glob.glob(pattern, recursive=recursive))


def print_output_summary():
    """Cảnh báo nếu output cũ làm số file không đúng kế hoạch."""
    print("\n== KIỂM TRA SỐ LƯỢNG OUTPUT ==")
    for _, dst_name in COMMANDS.items():
        count = count_wav_files(os.path.join(OUT_DIR, dst_name))
        status = "OK" if count == N_COMMAND else "CẦN KIỂM TRA"
        print(f"  {dst_name:10s}: {count:4d}/{N_COMMAND} file Google ({status})")

    kaggle_count = count_wav_files(BG_KAGGLE_OUT_DIR)
    own_count = count_wav_files(BG_OWN_OUT_DIR)
    print(f"  Background/Kaggle : {kaggle_count:4d}/{N_BG_KAGGLE}")
    print(f"  Background/INMP441: {own_count:4d}/{N_BG_OWN}")
    print(f"  Background tổng   : {kaggle_count + own_count:4d}/{N_BG_TOTAL}")
    print("Lưu ý: file cũ dư thừa không bị script tự động xóa.")


# ---------- Main ----------

def main():
    print("== BƯỚC 1: bốc file lệnh on/off/up/down ==")
    copy_random_commands()

    print("\n== BƯỚC 2: cắt background từ Kaggle (_background_noise_) ==")
    cut_random_segments(BG_KAGGLE_DIR, BG_KAGGLE_OUT_DIR, N_BG_KAGGLE, prefix="kaggle_bg")

    print("\n== BƯỚC 3: cắt background từ 3 file tự thu (tuthu / INMP441) ==")
    cut_random_segments(BG_OWN_DIR, BG_OWN_OUT_DIR, N_BG_OWN, prefix="own_bg")

    print_output_summary()
    print("\nHOÀN TẤT. Kết quả tại:", OUT_DIR)


if __name__ == "__main__":
    main()
