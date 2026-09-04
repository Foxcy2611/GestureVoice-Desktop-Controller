"""
====================
Tên script: Desktop_Action_Executor.py
Tác dụng: Ánh xạ DesktopCommand sang hotkey, media command, cửa sổ và brightness Windows.
====================
"""

"""Windows desktop action executor with a safe dry-run mode.

No action is performed unless Execute_Command(..., dry_run=False) is used.
The state machine is responsible for validation; this module still validates
the command-to-step mapping so malformed dictionaries never reach the OS.
"""

import ctypes
import os
import subprocess
import time


KEY_CODES = {
    "ALT": 0x12,
    "CTRL": 0x11,
    "SHIFT": 0x10,
    "TAB": 0x09,
    "ESC": 0x1B,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "F5": 0x74,
    "F11": 0x7A,
    "0": 0x30,
    "T": 0x54,
    "W": 0x57,
    "PLUS": 0xBB,
    "MINUS": 0xBD,
}

APP_COMMANDS = {
    "volume_mute": 8,
    "volume_down": 9,
    "volume_up": 10,
    "media_next": 11,
    "media_previous": 12,
    "media_stop": 13,
    "media_play_pause": 14,
    "media_play": 46,
    "media_pause": 47,
}

KEYEVENTF_KEYUP = 0x0002
WM_APPCOMMAND = 0x0319
HWND_BROADCAST = 0xFFFF
SW_MINIMIZE = 6
SW_MAXIMIZE = 3

ACTION_DESCRIPTIONS = {
    "slideshow": {
        "ON": "Bắt đầu trình chiếu (F5)",
        "OFF": "Thoát trình chiếu (Esc)",
        "UP": "Chuyển sang slide kế",
        "DOWN": "Quay về slide trước",
    },
    "volume": {
        "ON": "Bỏ mute và tăng âm lượng một nấc",
        "OFF": "Tắt tiếng toàn hệ thống; media vẫn tiếp tục phát",
        "UP": "Tăng âm lượng hệ thống",
        "DOWN": "Giảm âm lượng hệ thống",
    },
    "zoom": {
        "ON": "Đặt zoom về 100%",
        "UP": "Phóng to nội dung",
        "DOWN": "Thu nhỏ nội dung",
    },
    "fullscreen": {
        "ON": "Bật chế độ toàn màn hình",
        "OFF": "Thoát chế độ toàn màn hình",
    },
    "browser_tabs": {
        "ON": "Mở lại tab vừa đóng",
        "OFF": "Đóng tab hiện tại",
        "UP": "Chuyển sang tab kế",
        "DOWN": "Chuyển về tab trước",
    },
    "window": {
        "ON": "Phóng to cửa sổ đang active",
        "OFF": "Thu nhỏ cửa sổ đang active",
        "UP": "Chuyển sang ứng dụng kế",
        "DOWN": "Chuyển về ứng dụng trước",
    },
    "media": {
        "ON": "Phát media",
        "OFF": "Tạm dừng media",
        "UP": "Chuyển sang bài kế",
        "DOWN": "Quay về bài trước",
    },
    "browser_scroll": {
        "UP": "Cuộn trang lên",
        "DOWN": "Cuộn trang xuống",
    },
    "brightness": {
        "ON": "Đặt độ sáng màn hình về 70%",
        "OFF": "Đặt độ sáng màn hình về 10%",
        "UP": "Tăng độ sáng màn hình 10%",
        "DOWN": "Giảm độ sáng màn hình 10%",
    },
}


def Key_Step(*keys):
    return {"kind": "hotkey", "keys": list(keys)}


def App_Command_Step(name):
    return {"kind": "app_command", "name": name}


def Brightness_Step(operation, value):
    return {"kind": "brightness", "operation": operation, "value": int(value)}


def Window_Step(operation):
    return {"kind": "window", "operation": operation}


TARGET_ACTION_STEPS = {
    "slideshow": {
        "ON": [Key_Step("F5")],
        "OFF": [Key_Step("ESC")],
        "UP": [Key_Step("PAGEDOWN")],
        "DOWN": [Key_Step("PAGEUP")],
    },
    "volume": {
        # Volume-up reliably leaves Windows in an unmuted state. For OFF, doing
        # it before mute avoids accidentally unmuting an already muted endpoint.
        "ON": [App_Command_Step("volume_up")],
        "OFF": [App_Command_Step("volume_up"), App_Command_Step("volume_mute")],
        "UP": [App_Command_Step("volume_up")],
        "DOWN": [App_Command_Step("volume_down")],
    },
    "zoom": {
        "ON": [Key_Step("CTRL", "0")],
        "UP": [Key_Step("CTRL", "PLUS")],
        "DOWN": [Key_Step("CTRL", "MINUS")],
    },
    "fullscreen": {
        "ON": [Key_Step("F11")],
        "OFF": [Key_Step("ESC")],
    },
    "browser_tabs": {
        "ON": [Key_Step("CTRL", "SHIFT", "T")],
        "OFF": [Key_Step("CTRL", "W")],
        "UP": [Key_Step("CTRL", "TAB")],
        "DOWN": [Key_Step("CTRL", "SHIFT", "TAB")],
    },
    "window": {
        "ON": [Window_Step("maximize")],
        "OFF": [Window_Step("minimize")],
        "UP": [Key_Step("ALT", "TAB")],
        "DOWN": [Key_Step("ALT", "SHIFT", "TAB")],
    },
    "media": {
        "ON": [App_Command_Step("media_play")],
        "OFF": [App_Command_Step("media_pause")],
        "UP": [App_Command_Step("media_next")],
        "DOWN": [App_Command_Step("media_previous")],
    },
    "browser_scroll": {
        "UP": [Key_Step("PAGEUP")],
        "DOWN": [Key_Step("PAGEDOWN")],
    },
    "brightness": {
        "ON": [Brightness_Step("set", 70)],
        "OFF": [Brightness_Step("set", 10)],
        "UP": [Brightness_Step("change", 10)],
        "DOWN": [Brightness_Step("change", -10)],
    },
}


def Build_Action_Steps(command):
    if not isinstance(command, dict):
        return None, "command_not_dict"

    command_type = command.get("command_type")
    if command_type == "PANIC_STOP":
        return [
            Key_Step("ESC"),
            App_Command_Step("media_pause"),
            App_Command_Step("volume_up"),
            App_Command_Step("volume_mute"),
        ], None
    if command_type != "DESKTOP_ACTION":
        return None, "invalid_command_type"

    mode = command.get("mode")
    target = command.get("target")
    action = command.get("action")
    if mode not in {"presentation", "desktop_media"}:
        return None, "invalid_mode"

    if target == "profile":
        if action not in {"ON", "OFF"}:
            return None, "action_not_mapped"
        if mode == "presentation":
            return ([Key_Step("F5")] if action == "ON" else [Key_Step("ESC")]), None
        return (
            [App_Command_Step("media_play")]
            if action == "ON"
            else [App_Command_Step("media_pause")]
        ), None

    steps = TARGET_ACTION_STEPS.get(target, {}).get(action)
    if not steps:
        return None, "action_not_mapped"
    return [dict(step) for step in steps], None


def Describe_Command(command):
    """Trả mô tả tiếng Việt cho command để runtime/log không chỉ in dict."""
    if not isinstance(command, dict):
        return "Command không hợp lệ"
    if command.get("command_type") == "PANIC_STOP":
        return "Panic stop: thoát màn hình hiện tại, pause media và mute âm lượng"

    mode = command.get("mode")
    target = command.get("target")
    action = command.get("action")
    if target == "profile":
        if mode == "presentation":
            return "Bắt đầu presentation profile" if action == "ON" else "Dừng presentation profile"
        if mode == "desktop_media":
            return "Phát media profile" if action == "ON" else "Tạm dừng media profile"
    return ACTION_DESCRIPTIONS.get(target, {}).get(action, "Không có mô tả hành động")


def Format_Action_Steps(steps):
    """Định dạng danh sách primitive step thành một dòng dễ đọc."""
    formatted = []
    for step in steps or []:
        kind = step.get("kind")
        if kind == "hotkey":
            formatted.append("+".join(step.get("keys", [])))
        elif kind == "app_command":
            formatted.append(str(step.get("name")))
        elif kind == "window":
            formatted.append(f"window:{step.get('operation')}")
        elif kind == "brightness":
            formatted.append(
                f"brightness:{step.get('operation')}({step.get('value')})"
            )
        else:
            formatted.append(str(step))
    return " -> ".join(formatted)


def Press_Hotkey(keys):
    user32 = ctypes.windll.user32
    codes = []
    for key in keys:
        code = KEY_CODES.get(key)
        if code is None:
            raise ValueError(f"Unsupported key: {key}")
        codes.append(code)
    for code in codes:
        user32.keybd_event(code, 0, 0, 0)
    for code in reversed(codes):
        user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)


def Send_App_Command(name):
    command_id = APP_COMMANDS.get(name)
    if command_id is None:
        raise ValueError(f"Unsupported app command: {name}")
    # PostMessageW đưa lệnh vào message queue rồi trả về ngay. Bản cũ dùng
    # SendMessageW broadcast đồng bộ nên camera loop có thể đứng chờ một cửa
    # sổ khác xử lý message và Windows hiển thị "Not Responding".
    posted = ctypes.windll.user32.PostMessageW(
        HWND_BROADCAST,
        WM_APPCOMMAND,
        0,
        command_id << 16,
    )
    if not posted:
        raise ctypes.WinError()


def Change_Active_Window(operation):
    user32 = ctypes.windll.user32
    window = user32.GetForegroundWindow()
    if not window:
        raise RuntimeError("No foreground window")
    if operation == "maximize":
        user32.ShowWindow(window, SW_MAXIMIZE)
    elif operation == "minimize":
        user32.ShowWindow(window, SW_MINIMIZE)
    else:
        raise ValueError(f"Unsupported window operation: {operation}")


def Change_Brightness(operation, value):
    if operation == "set":
        target_expression = str(max(0, min(100, int(value))))
    elif operation == "change":
        delta = int(value)
        target_expression = (
            f"[Math]::Max(0,[Math]::Min(100,[int]$monitor.CurrentBrightness+({delta})))"
        )
    else:
        raise ValueError(f"Unsupported brightness operation: {operation}")

    script = (
        "$monitor=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
        "| Select-Object -First 1;"
        "$method=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods "
        "| Select-Object -First 1;"
        "if($null -eq $monitor -or $null -eq $method){throw 'Monitor brightness is unavailable'};"
        f"$target={target_expression};"
        "Invoke-CimMethod -InputObject $method -MethodName WmiSetBrightness "
        "-Arguments @{Timeout=1;Brightness=[byte]$target}|Out-Null"
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=8,
        creationflags=creation_flags,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "brightness command failed").strip()
        raise RuntimeError(message)


def Execute_Step(step):
    kind = step.get("kind")
    if kind == "hotkey":
        Press_Hotkey(step["keys"])
    elif kind == "app_command":
        Send_App_Command(step["name"])
    elif kind == "window":
        Change_Active_Window(step["operation"])
    elif kind == "brightness":
        Change_Brightness(step["operation"], step["value"])
    else:
        raise ValueError(f"Unsupported step kind: {kind}")


def Execute_Command(command, dry_run=True):
    steps, reason = Build_Action_Steps(command)
    if steps is None:
        return {
            "ok": False,
            "dry_run": bool(dry_run),
            "reason": reason,
            "steps": [],
        }
    if not dry_run and os.name != "nt":
        return {
            "ok": False,
            "dry_run": False,
            "reason": "windows_only_executor",
            "steps": steps,
        }

    executed = []
    if not dry_run:
        for step in steps:
            try:
                Execute_Step(step)
                executed.append(step)
                time.sleep(0.04)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                return {
                    "ok": False,
                    "dry_run": False,
                    "reason": "execution_failed",
                    "error": str(error),
                    "steps": steps,
                    "executed": executed,
                }

    return {
        "ok": True,
        "dry_run": bool(dry_run),
        "reason": None,
        "steps": steps,
        "executed": executed if not dry_run else [],
    }
