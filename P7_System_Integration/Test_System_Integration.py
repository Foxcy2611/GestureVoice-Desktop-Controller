"""
====================
Tên script: Test_System_Integration.py
Tác dụng: Kiểm thử end-to-end bằng event giả từ gesture/KWS đến executor dry-run.
====================
"""

"""Synthetic end-to-end tests from gesture events to dry-run OS steps."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from P5_Desktop_State_Machine import State_Machine_Core as state_core
from P6_Desktop_Action_Executor import Desktop_Action_Executor as executor


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


def Stable(machine, hand, label, start):
    outputs = []
    for index in range(state_core.GESTURE_STABLE_FRAMES):
        now = start + index * 0.03
        outputs.extend(state_core.Process_Gesture_Frame(machine, [{
            "hand": hand,
            "label": label,
            "confidence": 0.96,
            "timestamp": now,
        }], now))
    return outputs


def Voice(machine, action, timestamp):
    return state_core.Process_Voice_Event(machine, {
        "action": action,
        "confidence": 0.94,
        "timestamp": timestamp,
    })


def main():
    machine = state_core.Create_State_Machine()
    Stable(machine, "Left", "left_two", 1.0)
    Stable(machine, "Right", "right_one", 2.0)
    command = Voice(machine, "ON", 3.0)
    result = executor.Execute_Command(command, dry_run=True)
    Check(command["target"] == "media" and result["ok"] and "media_play" in str(result),
          "desktop -> media -> ON produces media_play")

    machine = state_core.Create_State_Machine()
    Stable(machine, "Left", "left_one", 5.0)
    Stable(machine, "Right", "right_one", 6.0)
    command = Voice(machine, "DOWN", 7.0)
    result = executor.Execute_Command(command, dry_run=True)
    Check(command["target"] == "slideshow" and "PAGEUP" in str(result),
          "presentation -> slideshow -> DOWN produces previous slide")

    machine = state_core.Create_State_Machine()
    Stable(machine, "Left", "left_two", 10.0)
    Stable(machine, "Right", "right_four", 11.0)
    command = Voice(machine, "UP", 12.0)
    result = executor.Execute_Command(command, dry_run=True)
    Check(command["target"] == "browser_tabs" and "TAB" in str(result),
          "desktop -> browser_tabs -> UP produces next tab")

    machine = state_core.Create_State_Machine()
    Stable(machine, "Left", "left_two", 20.0)
    Stable(machine, "Right", "right_three", 21.0)
    rejected = Voice(machine, "ON", 22.0)
    Check(rejected["event_type"] == "REJECTED" and rejected["reason"] == "action_not_supported",
          "invalid capability stops before executor")

    machine = state_core.Create_State_Machine()
    outputs = Stable(machine, "Right", "right_ok", 30.0)
    result = executor.Execute_Command(outputs[-1], dry_run=True)
    Check(outputs[-1]["command_type"] == "PANIC_STOP" and len(result["steps"]) == 4,
          "panic works without mode, target, or keyword")

    print(f"\nPhase 7 result: {passed}/{total} PASS")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
