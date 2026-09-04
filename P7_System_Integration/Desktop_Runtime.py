"""
====================
Tên script: Desktop_Runtime.py
Tác dụng: Chạy end-to-end webcam + micro + State Machine + Desktop Executor.
Run: .\venv\Scripts\python.exe .\P7_System_Integration\Desktop_Runtime.py --execute
====================
"""

"""End-to-end webcam + microphone runtime for GestureVoice Desktop Controller.

The default is DRY-RUN: recognized commands are printed but no keyboard,
media, window, or brightness action is sent. Pass --execute explicitly to
enable local operating-system actions.
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from P4_Real_Time_Evaluation import Eval_Gesture_Webcam as gesture_ai
from P4_Real_Time_Evaluation import Eval_KWS_Mic as kws_ai
from P5_Desktop_State_Machine import State_Machine_Core as state_core
from P6_Desktop_Action_Executor import Desktop_Action_Executor as desktop_executor


MODE_GESTURE_BY_MODE = {
    mode: f"left_{gesture}"
    for gesture, mode in state_core.MODE_BY_LEFT_GESTURE.items()
}
TARGET_GESTURE_BY_MODE = {
    mode: {target: f"right_{gesture}" for gesture, target in targets.items()}
    for mode, targets in state_core.TARGET_BY_MODE.items()
}
ACTION_ORDER = ("ON", "OFF", "UP", "DOWN")


def Poll_Voice_Event(model, classes, source_samplerate, energy_gate_rms, vad_threshold):
    audio = kws_ai.Read_Last_1s(kws_ai.RAW_CAPTURE_SEC)
    if audio is None:
        return None

    rms = kws_ai.Calculate_RMS(audio, sr=source_samplerate)
    if rms < energy_gate_rms:
        kws_ai.Reset_Predict_History()
        kws_ai.Reset_Output_State()
        return None
    if kws_ai.Is_In_Command_Cooldown():
        kws_ai.Reset_Predict_History()
        return None

    feature = kws_ai.Processing_Number(
        audio,
        source_samplerate,
        vad_threshold=vad_threshold,
    )
    label, confidence, _ = kws_ai.Predict_Window(model, classes, feature)
    if label == "background":
        kws_ai.Reset_Predict_History()
        return None

    voted_label, _, voted_confidence = kws_ai.Add_History_and_Vote(label, confidence)
    if voted_label is None or voted_confidence < kws_ai.CONFIDENCE_THRESHOLD:
        return None
    if not kws_ai.Should_Print_Command():
        return None

    kws_ai.Reset_Predict_History()
    return {
        "action": voted_label.upper(),
        "confidence": float(voted_confidence),
        "timestamp": time.monotonic(),
    }


def Calibrate_Microphone(source_samplerate):
    print(
        f"Calibrating background noise for {kws_ai.CALIBRATION_SEC:.1f}s; "
        "please remain quiet..."
    )
    time.sleep(kws_ai.CALIBRATION_SEC)
    audio = kws_ai.Read_Last_1s(kws_ai.CALIBRATION_SEC)
    while audio is None:
        time.sleep(0.05)
        audio = kws_ai.Read_Last_1s(kws_ai.CALIBRATION_SEC)

    energy_gate_rms, noise_rms = kws_ai.Calibrate_Energy_Gate(audio, source_samplerate)
    vad_threshold = float(kws_ai.np.clip(
        noise_rms * kws_ai.VAD_NOISE_MULTIPLIER,
        kws_ai.MIN_VAD_RMS,
        kws_ai.MAX_VAD_RMS,
    ))
    print(f"Noise RMS={noise_rms:.6f}; gate={energy_gate_rms:.6f}; VAD={vad_threshold:.6f}")
    return energy_gate_rms, vad_threshold


def Format_Selection(machine):
    selection = state_core.Get_Selection_Snapshot(machine)
    return (
        f"Mode: {selection.get('mode') or '-'} | "
        f"Target: {selection.get('target') or '-'}"
    )


def Format_Combination(event):
    if event.get("command_type") == "PANIC_STOP":
        return "right_ok"
    mode = event.get("mode")
    target = event.get("target")
    left_gesture = MODE_GESTURE_BY_MODE.get(mode, "left_?")
    right_gesture = TARGET_GESTURE_BY_MODE.get(mode, {}).get(target, "right_?")
    return f"{left_gesture} + {right_gesture} + {event.get('action')}"


def Handle_Output(event, execute_actions=False):
    if event is None:
        return ""
    if event.get("event_type") == "DESKTOP_COMMAND":
        result = desktop_executor.Execute_Command(event, dry_run=not execute_actions)
        mode_text = "EXECUTE" if execute_actions else "DRY-RUN"
        combination = Format_Combination(event)
        description = desktop_executor.Describe_Command(event)
        steps_text = desktop_executor.Format_Action_Steps(result.get("steps"))
        print(f"[FLOW 3/3] {combination}")
        print(
            f"[COMMAND] mode={event.get('mode') or '-'} | "
            f"target={event.get('target')} | action={event.get('action') or 'STOP'}"
        )
        print(f"[ACTION] {description}")
        if result["ok"]:
            result_word = "ĐÃ THỰC THI" if execute_actions else "CHỈ MÔ PHỎNG"
            print(f"[RESULT/{mode_text}] {result_word} | steps: {steps_text}")
            print("-" * 72)
            return f"{mode_text}: {event.get('target')} {event.get('action') or 'STOP'}"
        print(
            f"[RESULT/{mode_text}] THẤT BẠI | reason={result.get('reason')} | "
            f"error={result.get('error', '-') }"
        )
        print("-" * 72)
        return f"EXECUTOR ERROR: {result.get('reason')}"
    if event.get("event_type") == "REJECTED":
        print(
            f"[REJECTED] source={event.get('source')} | "
            f"reason={event.get('reason')}"
        )
        return f"REJECTED: {event.get('reason')}"

    mode = event.get("mode")
    target = event.get("target")
    source_gesture = event.get("gesture", "?")
    if target is None:
        print(f"[FLOW 1/3] {source_gesture} -> mode={mode}")
        print("[NEXT] Giơ gesture tay phải để chọn chức năng.")
    else:
        left_gesture = MODE_GESTURE_BY_MODE.get(mode, "left_?")
        supported = state_core.CAPABILITIES.get(mode, {}).get(target, set())
        supported_text = "/".join(action for action in ACTION_ORDER if action in supported)
        print(f"[FLOW 2/3] {left_gesture} + {source_gesture} -> target={target}")
        print(f"[NEXT] Nói keyword hợp lệ cho target này: {supported_text}")
    return f"SELECTED: {event.get('mode')} / {event.get('target') or '-'}"


def Check_Models():
    gesture_model, gesture_classes = gesture_ai.Load_Model_And_Classes()
    kws_model, kws_classes = kws_ai.Load_Model_and_Label()
    print(
        "Gesture model:",
        gesture_model.input_shape,
        "->",
        gesture_model.output_shape,
        list(gesture_classes),
    )
    print(
        "KWS model:",
        kws_model.input_shape,
        "->",
        kws_model.output_shape,
        list(kws_classes),
    )


def Run_Runtime(execute_actions=False, camera_index=0):
    print("Loading gesture and KWS models...")
    gesture_model, gesture_classes = gesture_ai.Load_Model_And_Classes()
    kws_model, kws_classes = kws_ai.Load_Model_and_Label()

    input_device, mic_info = kws_ai.Select_Input_Device()
    source_samplerate = int(mic_info["default_samplerate"])
    host_api = kws_ai.sd.query_hostapis(int(mic_info["hostapi"]))["name"]
    print(f"Microphone: [{input_device}] {mic_info['name']} ({host_api})")
    print(f"Microphone sample rate: {source_samplerate} Hz")

    kws_ai.Init_Buffer(
        max_seconds=max(kws_ai.CALIBRATION_SEC + 0.5, kws_ai.RAW_CAPTURE_SEC + 0.5),
        samplerate=source_samplerate,
    )
    kws_ai.Reset_Predict_History()
    kws_ai.Reset_Output_State()

    mp_hands = gesture_ai.mp.solutions.hands
    mp_draw = gesture_ai.mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    camera = gesture_ai.cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        hands.close()
        raise RuntimeError(f"Cannot open webcam index {camera_index}")

    machine = state_core.Create_State_Machine()
    last_output_text = ""
    last_output_time = 0.0
    next_audio_poll = 0.0
    previous_frame_time = time.monotonic()

    mode_text = "EXECUTE" if execute_actions else "DRY-RUN"
    print(f"Runtime mode: {mode_text}")
    print("Order: left gesture -> right gesture -> say ON/OFF/UP/DOWN")
    print("right_ok triggers PANIC_STOP without voice; press q to quit.\n")

    try:
        with kws_ai.sd.InputStream(
            device=input_device,
            samplerate=source_samplerate,
            channels=1,
            dtype="float32",
            callback=kws_ai.Record_Callbacks,
        ):
            energy_gate_rms, vad_threshold = Calibrate_Microphone(source_samplerate)
            while True:
                frame_ok, frame = camera.read()
                if not frame_ok:
                    raise RuntimeError("Cannot read webcam frame")

                now = time.monotonic()
                frame = gesture_ai.cv2.flip(frame, 1)
                rgb = gesture_ai.cv2.cvtColor(frame, gesture_ai.cv2.COLOR_BGR2RGB)
                detection = hands.process(rgb)
                raw_gestures = []

                if detection.multi_hand_landmarks:
                    for landmarks, handedness in zip(
                        detection.multi_hand_landmarks,
                        detection.multi_handedness,
                    ):
                        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                        hand = handedness.classification[0].label
                        feature = gesture_ai.Normalize_Landmarks(landmarks)
                        label, confidence = gesture_ai.Predict_Gesture(
                            gesture_model,
                            gesture_classes,
                            feature,
                        )
                        raw_gestures.append({
                            "hand": hand,
                            "label": label,
                            "confidence": float(confidence),
                            "timestamp": now,
                        })

                        height, width, _ = frame.shape
                        wrist = (
                            int(landmarks.landmark[0].x * width),
                            int(landmarks.landmark[0].y * height) - 20,
                        )
                        mismatch = hand.lower() not in label.lower()
                        color = (0, 0, 255) if mismatch else (0, 255, 0)
                        if confidence < state_core.GESTURE_CONFIDENCE_THRESHOLD:
                            color = (0, 165, 255)
                        gesture_ai.cv2.putText(
                            frame,
                            f"{label} {confidence * 100:.1f}% [{hand}]",
                            wrist,
                            gesture_ai.cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                for output in state_core.Process_Gesture_Frame(machine, raw_gestures, now):
                    text = Handle_Output(output, execute_actions)
                    if text:
                        last_output_text = text
                        last_output_time = now

                if now >= next_audio_poll:
                    next_audio_poll = now + kws_ai.HOP_SEC
                    voice_event = Poll_Voice_Event(
                        kws_model,
                        kws_classes,
                        source_samplerate,
                        energy_gate_rms,
                        vad_threshold,
                    )
                    if voice_event is not None:
                        print(
                            f"[KWS] keyword={voice_event['action']} | "
                            f"confidence={voice_event['confidence'] * 100:.1f}%"
                        )
                        output = state_core.Process_Voice_Event(machine, voice_event)
                        text = Handle_Output(output, execute_actions)
                        if text:
                            last_output_text = text
                            last_output_time = now

                frame_delta = now - previous_frame_time
                fps = 1.0 / frame_delta if frame_delta > 0 else 0.0
                previous_frame_time = now
                gesture_ai.cv2.putText(
                    frame,
                    Format_Selection(machine),
                    (10, 30),
                    gesture_ai.cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )
                gesture_ai.cv2.putText(
                    frame,
                    f"{mode_text} | FPS: {fps:.1f}",
                    (10, 58),
                    gesture_ai.cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255) if not execute_actions else (0, 0, 255),
                    2,
                )
                if last_output_text and now - last_output_time <= 3.0:
                    gesture_ai.cv2.putText(
                        frame,
                        last_output_text,
                        (10, 88),
                        gesture_ai.cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 255, 255),
                        2,
                    )

                gesture_ai.cv2.imshow("GestureVoice Desktop Controller", frame)
                if gesture_ai.cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        hands.close()
        gesture_ai.cv2.destroyAllWindows()


def Parse_Arguments():
    parser = argparse.ArgumentParser(description="Gesture + KWS desktop controller")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform real Windows actions; default only prints dry-run steps",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="load both models and exit without opening webcam or microphone",
    )
    return parser.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = Parse_Arguments()
    try:
        if args.check_models:
            Check_Models()
            return
        Run_Runtime(execute_actions=args.execute, camera_index=args.camera_index)
    except KeyboardInterrupt:
        print("\nRuntime stopped.")


if __name__ == "__main__":
    main()
