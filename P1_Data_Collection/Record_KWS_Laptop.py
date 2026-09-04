"""
====================
Tên script: Record_KWS_Laptop.py
Tác dụng: Thu bổ sung mẫu ON/OFF/UP/DOWN và background bằng microphone laptop.
====================
"""

"""
Thu tối đa 30 mẫu cá nhân/lớp bằng microphone laptop.

Mỗi file dài đúng 1 giây và được lưu vào:
    Dataset/Voice_Data_Raw/On
    Dataset/Voice_Data_Raw/Off
    Dataset/Voice_Data_Raw/Up
    Dataset/Voice_Data_Raw/Down
    Dataset/Voice_Data_Raw/Background/Laptop

Script có thể chạy lại sau khi bị gián đoạn: nó chỉ thu số mẫu
"laptop_<label>_*.wav" còn thiếu để đạt --count (mặc định 30).
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


for output_stream in (sys.stdout, sys.stderr):
    if hasattr(output_stream, "reconfigure"):
        output_stream.reconfigure(encoding="utf-8", errors="replace")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset", "Voice_Data_Raw")

LABEL_DIRS = {
    "on": os.path.join(DATASET_DIR, "On"),
    "off": os.path.join(DATASET_DIR, "Off"),
    "up": os.path.join(DATASET_DIR, "Up"),
    "down": os.path.join(DATASET_DIR, "Down"),
    "background": os.path.join(DATASET_DIR, "Background", "Laptop"),
}

KEYWORD_ORDER = ["on", "off", "up", "down", "background"]
DEFAULT_COUNT = 30
DEFAULT_DURATION_SEC = 1.0
DEFAULT_COUNTDOWN_SEC = 1.0
DEFAULT_INTERVAL_SEC = 0.5
MIN_VALID_PEAK = 1e-7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thu 30 mẫu/lớp KWS bằng microphone laptop."
    )
    parser.add_argument(
        "--label",
        choices=KEYWORD_ORDER + ["all"],
        default="all",
        help="Lớp cần thu; mặc định thu lần lượt tất cả.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Tổng số file laptop mong muốn trong mỗi lớp.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Thời lượng mỗi file, tính bằng giây.",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=DEFAULT_COUNTDOWN_SEC,
        help="Thời gian chuẩn bị trước mỗi lần thu.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help="Khoảng nghỉ sau mỗi lần thu.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="SoundDevice input index; bỏ trống để tự ưu tiên WASAPI.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="In danh sách thiết bị input rồi thoát.",
    )
    return parser.parse_args()


def list_input_devices():
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) < 1:
            continue
        host_api = sd.query_hostapis(int(device["hostapi"]))["name"]
        print(
            f"[{index:2d}] {host_api:20s} "
            f"sr={int(device['default_samplerate']):5d}  {device['name']}"
        )


def select_input_device(requested_device=None):
    if requested_device is not None:
        info = sd.query_devices(requested_device)
        if int(info["max_input_channels"]) < 1:
            raise ValueError(f"Device {requested_device} không phải thiết bị input.")
        return requested_device, info

    default_index = int(sd.default.device[0])
    default_info = sd.query_devices(default_index)

    if os.name == "nt":
        name_key = str(default_info["name"]).split("(", 1)[0].strip().lower()
        for index, device in enumerate(sd.query_devices()):
            if int(device["max_input_channels"]) < 1:
                continue
            host_api = sd.query_hostapis(int(device["hostapi"]))["name"]
            same_microphone = name_key and name_key in str(device["name"]).lower()
            if host_api == "Windows WASAPI" and same_microphone:
                return index, device

    return default_index, default_info


def existing_laptop_indices(label, target_dir):
    pattern = os.path.join(target_dir, f"laptop_{label}_*.wav")
    regex = re.compile(rf"laptop_{re.escape(label)}_(\d+)\.wav$", re.IGNORECASE)
    indices = []
    for path in glob.glob(pattern):
        match = regex.search(os.path.basename(path))
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def record_audio(device_index, samplerate, duration):
    sample_count = int(round(samplerate * duration))
    audio = sd.rec(
        sample_count,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    return audio[:, 0]


def record_label(label, target_count, device_index, samplerate, args):
    target_dir = LABEL_DIRS[label]
    os.makedirs(target_dir, exist_ok=True)

    existing_indices = existing_laptop_indices(label, target_dir)
    existing_count = len(existing_indices)
    remaining = max(0, target_count - existing_count)

    print(f"\n{'=' * 64}")
    print(f"Lớp: {label.upper()} | đã có {existing_count}/{target_count} mẫu laptop")
    print(f"Thư mục: {target_dir}")
    if remaining == 0:
        print("[SKIP] Lớp này đã đủ mẫu.")
        return

    if label == "background":
        instruction = "Không nói; để mic thu tiếng nền môi trường."
    else:
        instruction = f"Mỗi khi hiện 'THU', nói một lần từ: {label.upper()}"

    print(instruction)
    input(f"Nhấn Enter để bắt đầu thu {remaining} mẫu còn thiếu...")

    next_index = existing_indices[-1] + 1 if existing_indices else 0
    saved_now = 0
    while existing_count + saved_now < target_count:
        ordinal = existing_count + saved_now + 1
        print(f"\n[{ordinal:02d}/{target_count:02d}] Chuẩn bị...", flush=True)
        time.sleep(max(0.0, args.countdown))
        action = "GIỮ IM LẶNG" if label == "background" else f"THU: {label.upper()}"
        print(f"[{ordinal:02d}/{target_count:02d}] {action}", flush=True)

        audio = record_audio(device_index, samplerate, args.duration)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if len(audio) else 0.0

        if peak <= MIN_VALID_PEAK:
            print("[RETRY] Tín hiệu rỗng/quá thấp; kiểm tra mic và thu lại mẫu này.")
            continue

        filename = f"laptop_{label}_{next_index:04d}.wav"
        output_path = os.path.join(target_dir, filename)
        # WAV float32 giữ được tín hiệu rất nhỏ của microphone array.
        sf.write(output_path, audio, samplerate, subtype="FLOAT")
        print(f"[OK] {filename} | rms={rms:.6f} peak={peak:.6f}")

        next_index += 1
        saved_now += 1
        time.sleep(max(0.0, args.interval))

    print(f"[DONE] {label.upper()}: đã đủ {target_count} mẫu laptop.")


def main():
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    if args.count <= 0:
        raise ValueError("--count phải lớn hơn 0.")
    if args.duration <= 0:
        raise ValueError("--duration phải lớn hơn 0.")

    device_index, device_info = select_input_device(args.device)
    samplerate = int(device_info["default_samplerate"])
    host_api = sd.query_hostapis(int(device_info["hostapi"]))["name"]
    print(f"Mic: [{device_index}] {device_info['name']} ({host_api})")
    print(f"Sample rate: {samplerate} Hz | Mỗi file: {args.duration:.2f}s")
    print("Có thể nhấn Ctrl+C để dừng; các file đã thu vẫn được giữ lại.")

    labels = KEYWORD_ORDER if args.label == "all" else [args.label]
    try:
        for label in labels:
            record_label(label, args.count, device_index, samplerate, args)
    except KeyboardInterrupt:
        print("\nĐã dừng thu theo yêu cầu.")
        return

    print("\nHOÀN TẤT THU DỮ LIỆU LAPTOP.")


if __name__ == "__main__":
    main()
