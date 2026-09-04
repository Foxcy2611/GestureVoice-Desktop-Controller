"""
====================
Tên script: Test_State_Machine.py
Tác dụng: Kiểm thử offline mapping, debounce, timeout, capability và panic của Phase 5.
====================
"""

"""Offline tests for Phase 5; no model, webcam, microphone, or OS actions."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from P5_Desktop_State_Machine import State_Machine_Core as core


passed = 0
total = 0


def Check(condition, name):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")


def Stable_Gesture(machine, hand, label, start_time):
    outputs = []
    for index in range(core.GESTURE_STABLE_FRAMES):
        timestamp = start_time + index * 0.03
        outputs.extend(core.Process_Gesture_Frame(machine, [{
            "hand": hand,
            "label": label,
            "confidence": 0.95,
            "timestamp": timestamp,
        }], timestamp))
    return outputs


def Voice(action, timestamp, confidence=0.95):
    return {"action": action, "confidence": confidence, "timestamp": timestamp}


def main():
    machine = core.Create_State_Machine()

    outputs = Stable_Gesture(machine, "Left", "left_one", 1.0)
    Check(len(outputs) == 1 and outputs[0]["mode"] == "presentation",
          "left_one selects presentation mode")
    Check(outputs[0].get("gesture") == "left_one",
          "mode SelectionEvent keeps source gesture")

    outputs = Stable_Gesture(machine, "Right", "right_one", 2.0)
    Check(len(outputs) == 1 and outputs[0]["target"] == "slideshow",
          "right_one selects slideshow in presentation mode")
    Check(outputs[0].get("gesture") == "right_one",
          "target SelectionEvent keeps source gesture")

    command = core.Process_Voice_Event(machine, Voice("UP", 3.0))
    Check(command and command["target"] == "slideshow" and command["action"] == "UP",
          "UP creates slideshow command")

    command = core.Process_Voice_Event(machine, Voice("DOWN", 3.1))
    Check(command and command["action"] == "DOWN", "DOWN is a different command")

    invalid = core.Process_Voice_Event(machine, Voice("NEXT", 3.2))
    Check(invalid is None, "keywords outside ON/OFF/UP/DOWN are rejected")

    machine = core.Create_State_Machine()
    outputs = Stable_Gesture(machine, "Right", "right_three", 4.0)
    Check(outputs[-1]["reason"] == "mode_not_selected",
          "right gesture before mode is rejected")

    rejected = core.Process_Voice_Event(machine, Voice("ON", 5.0))
    Check(rejected["reason"] == "mode_not_selected", "voice before mode is rejected")

    machine = core.Create_State_Machine()
    Stable_Gesture(machine, "Left", "left_two", 6.0)
    outputs = Stable_Gesture(machine, "Right", "right_three", 7.0)
    Check(outputs[-1]["target"] == "browser_scroll",
          "right_three maps to browser_scroll in desktop mode")

    rejected = core.Process_Voice_Event(machine, Voice("ON", 8.0))
    Check(rejected["reason"] == "action_not_supported",
          "browser_scroll rejects ON")

    command = core.Process_Voice_Event(machine, Voice("DOWN", 8.1))
    Check(command and command["target"] == "browser_scroll",
          "browser_scroll accepts DOWN")

    Stable_Gesture(machine, "Left", "left_one", 9.0)
    snapshot = core.Get_Selection_Snapshot(machine)
    Check(snapshot["mode"] == "presentation" and snapshot["target"] is None,
          "changing mode clears target")

    outputs = Stable_Gesture(machine, "Right", "right_like", 10.0)
    Check(outputs[-1]["target"] == "profile", "right_like selects profile")

    command = core.Process_Voice_Event(machine, Voice("ON", 11.0))
    Check(command and command["target"] == "profile", "profile accepts ON")

    machine = core.Create_State_Machine()
    outputs = Stable_Gesture(machine, "Right", "right_ok", 12.0)
    Check(outputs[-1]["command_type"] == "PANIC_STOP",
          "right_ok creates immediate panic command")

    machine = core.Create_State_Machine()
    Stable_Gesture(machine, "Left", "left_one", 20.0)
    Stable_Gesture(machine, "Right", "right_one", 21.0)
    rejected = core.Process_Voice_Event(machine, Voice("ON", 34.0))
    Check(rejected["reason"] == "selection_timeout", "selection expires after 12 seconds")

    machine = core.Create_State_Machine()
    Stable_Gesture(machine, "Left", "left_two", 40.0)
    Stable_Gesture(machine, "Right", "right_one", 41.0)
    first = core.Process_Voice_Event(machine, Voice("ON", 42.0))
    duplicate = core.Process_Voice_Event(machine, Voice("ON", 42.2))
    Check(first is not None and duplicate is None, "identical command is deduplicated")

    machine = core.Create_State_Machine()
    for index in range(4):
        outputs = core.Process_Gesture_Frame(machine, [{
            "hand": "Left", "label": "left_one", "confidence": 0.95,
        }], 50.0 + index * 0.03)
        Check(outputs == [], f"debounce frame {index + 1} has no output")
    outputs = core.Process_Gesture_Frame(machine, [{
        "hand": "Left", "label": "left_one", "confidence": 0.95,
    }], 50.12)
    Check(len(outputs) == 1, "fifth stable frame confirms gesture")

    outputs = core.Process_Gesture_Frame(machine, [{
        "hand": "Left", "label": "left_one", "confidence": 0.95,
    }], 50.15)
    Check(outputs == [], "held gesture does not repeat")

    mismatch = core.Normalize_Gesture_Label("Right", "left_one")
    Check(mismatch is None, "handedness mismatch is rejected")

    print(f"\nPhase 5 result: {passed}/{total} PASS")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
