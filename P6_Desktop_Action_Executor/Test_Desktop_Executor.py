"""
====================
Tên script: Test_Desktop_Executor.py
Tác dụng: Kiểm thử toàn bộ mapping executor ở dry-run, không gửi thao tác thật vào Windows.
====================
"""

"""Dry-run tests for Phase 6. The test never sends input to Windows."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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


def Command(mode, target, action):
    return {
        "event_type": "DESKTOP_COMMAND",
        "command_type": "DESKTOP_ACTION",
        "mode": mode,
        "target": target,
        "action": action,
        "source": "test",
        "timestamp": 1.0,
    }


def main():
    cases = [
        ("presentation", "slideshow", "ON", "F5"),
        ("presentation", "slideshow", "UP", "PAGEDOWN"),
        ("presentation", "zoom", "DOWN", "MINUS"),
        ("presentation", "fullscreen", "OFF", "ESC"),
        ("presentation", "browser_tabs", "ON", "T"),
        ("presentation", "window", "ON", "maximize"),
        ("desktop_media", "media", "OFF", "media_pause"),
        ("desktop_media", "volume", "UP", "volume_up"),
        ("desktop_media", "browser_scroll", "DOWN", "PAGEDOWN"),
        ("desktop_media", "browser_tabs", "UP", "TAB"),
        ("desktop_media", "brightness", "ON", 70),
        ("desktop_media", "window", "OFF", "minimize"),
    ]
    for mode, target, action, expected in cases:
        result = executor.Execute_Command(Command(mode, target, action), dry_run=True)
        Check(result["ok"] and str(expected) in str(result["steps"]),
              f"{mode}/{target}/{action} maps to {expected}")

    result = executor.Execute_Command(Command("presentation", "profile", "ON"), True)
    Check(result["ok"] and "F5" in str(result["steps"]),
          "presentation profile ON starts slideshow")

    result = executor.Execute_Command(Command("desktop_media", "profile", "OFF"), True)
    Check(result["ok"] and "media_pause" in str(result["steps"]),
          "desktop profile OFF pauses media")

    panic = {
        "event_type": "DESKTOP_COMMAND",
        "command_type": "PANIC_STOP",
        "mode": None,
        "target": "panic",
        "action": None,
        "source": "test",
        "timestamp": 2.0,
    }
    result = executor.Execute_Command(panic, True)
    Check(result["ok"] and len(result["steps"]) == 4, "panic maps to safe action sequence")

    result = executor.Execute_Command(Command("desktop_media", "browser_scroll", "ON"), True)
    Check(not result["ok"] and result["reason"] == "action_not_mapped",
          "unsupported action is rejected")

    result = executor.Execute_Command({"command_type": "UNKNOWN"}, True)
    Check(not result["ok"], "unknown command type is rejected")

    volume_off = Command("presentation", "volume", "OFF")
    Check("Tắt tiếng toàn hệ thống" in executor.Describe_Command(volume_off),
          "volume OFF has a human-readable description")

    steps, _ = executor.Build_Action_Steps(volume_off)
    Check(executor.Format_Action_Steps(steps) == "volume_up -> volume_mute",
          "executor steps have a compact readable format")

    print(f"\nPhase 6 result: {passed}/{total} PASS")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
