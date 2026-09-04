"""
====================
Tên script: State_Machine_Core.py
Tác dụng: Debounce gesture, giữ selection và ghép bốn KWS action thành DesktopCommand.
====================
"""

"""Pure state machine that fuses debounced gestures with four KWS actions.

The module has no camera, microphone, UI, MQTT, or operating-system calls.
It accepts plain dictionaries and returns SelectionEvent, RejectedEvent, or
DesktopCommand dictionaries so it can be tested without hardware.
"""

MODE_BY_LEFT_GESTURE = {
    "one": "presentation",
    "two": "desktop_media",
}

TARGET_BY_MODE = {
    "presentation": {
        "one": "slideshow",
        "two": "volume",
        "three": "zoom",
        "four": "fullscreen",
        "five": "browser_tabs",
        "fist": "window",
        "like": "profile",
    },
    "desktop_media": {
        "one": "media",
        "two": "volume",
        "three": "browser_scroll",
        "four": "browser_tabs",
        "five": "brightness",
        "fist": "window",
        "like": "profile",
    },
}

# Only these four words are commands. "background" remains an inference class
# in Phase 4 and is deliberately never forwarded to this core.
KWS_ACTIONS = {"ON", "OFF", "UP", "DOWN"}

CAPABILITIES = {
    "presentation": {
        "slideshow": KWS_ACTIONS,
        "volume": KWS_ACTIONS,
        "zoom": {"ON", "UP", "DOWN"},
        "fullscreen": {"ON", "OFF"},
        "browser_tabs": KWS_ACTIONS,
        "window": KWS_ACTIONS,
        "profile": {"ON", "OFF"},
    },
    "desktop_media": {
        "media": KWS_ACTIONS,
        "volume": KWS_ACTIONS,
        "browser_scroll": {"UP", "DOWN"},
        "browser_tabs": KWS_ACTIONS,
        "brightness": KWS_ACTIONS,
        "window": KWS_ACTIONS,
        "profile": {"ON", "OFF"},
    },
}

PANIC_GESTURE = "ok"
GESTURE_CONFIDENCE_THRESHOLD = 0.70
GESTURE_STABLE_FRAMES = 5
GESTURE_RELEASE_FRAMES = 5
GESTURE_MAX_FRAME_GAP_SEC = 0.25
SELECTION_TIMEOUT_SEC = 12.0
COMMAND_DEDUP_SEC = 0.75
VALID_HANDS = ("Left", "Right")


def Create_Hand_Debounce_State():
    return {
        "candidate_label": None,
        "stable_count": 0,
        "confidence_sum": 0.0,
        "last_frame_time": None,
        "missing_frames": 0,
        "locked_label": None,
    }


def Create_State_Machine():
    return {
        "gesture_debounce": {
            "Left": Create_Hand_Debounce_State(),
            "Right": Create_Hand_Debounce_State(),
        },
        "current_selection": {
            "mode": None,
            "target": None,
            "last_updated": None,
        },
        "last_command_signature": None,
        "last_command_time": None,
    }


def Get_Selection_Snapshot(machine):
    selection = machine["current_selection"]
    return {
        "mode": selection.get("mode"),
        "target": selection.get("target"),
        "last_updated": selection.get("last_updated"),
    }


def Reset_Selection(machine):
    selection = machine["current_selection"]
    selection["mode"] = None
    selection["target"] = None
    selection["last_updated"] = None


def Expire_Selection_If_Needed(machine, timestamp):
    last_updated = machine["current_selection"].get("last_updated")
    if isinstance(last_updated, bool) or not isinstance(last_updated, (int, float)):
        return False
    elapsed = float(timestamp) - float(last_updated)
    if elapsed < 0 or elapsed >= SELECTION_TIMEOUT_SEC:
        Reset_Selection(machine)
        return True
    return False


def Make_Selection_Event(selection, timestamp, gesture=None):
    event = {
        "event_type": "SELECTION",
        "mode": selection.get("mode"),
        "target": selection.get("target"),
        "timestamp": float(timestamp),
    }
    if gesture is not None:
        event["gesture"] = str(gesture)
    return event


def Make_Rejected_Event(source, reason, timestamp):
    return {
        "event_type": "REJECTED",
        "source": source,
        "reason": reason,
        "timestamp": float(timestamp),
    }


def Make_Command_Event(command_type, mode, target, action, source, timestamp):
    return {
        "event_type": "DESKTOP_COMMAND",
        "command_type": command_type,
        "mode": mode,
        "target": target,
        "action": action,
        "source": source,
        "timestamp": float(timestamp),
    }


def Normalize_Hand(hand):
    value = str(hand).strip().lower()
    if value == "left":
        return "Left"
    if value == "right":
        return "Right"
    return None


def Normalize_Gesture_Label(hand, label):
    value = str(label).strip().lower()
    if value.startswith("left_"):
        if hand != "Left":
            return None
        value = value[5:]
    elif value.startswith("right_"):
        if hand != "Right":
            return None
        value = value[6:]

    if hand == "Left":
        return value if value in MODE_BY_LEFT_GESTURE else None

    right_labels = set(TARGET_BY_MODE["presentation"])
    right_labels.update(TARGET_BY_MODE["desktop_media"])
    right_labels.add(PANIC_GESTURE)
    return value if value in right_labels else None


def Reset_Gesture_Candidate(hand_state):
    hand_state["candidate_label"] = None
    hand_state["stable_count"] = 0
    hand_state["confidence_sum"] = 0.0


def Mark_Hand_Missing(hand_state):
    hand_state["missing_frames"] += 1
    Reset_Gesture_Candidate(hand_state)
    if hand_state["missing_frames"] >= GESTURE_RELEASE_FRAMES:
        hand_state["locked_label"] = None
        hand_state["last_frame_time"] = None


def Debounce_One_Hand(hand_state, event, frame_timestamp):
    hand_state["missing_frames"] = 0
    label = event.get("label")
    confidence = event.get("confidence")
    if (
        label is None
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
        or confidence < GESTURE_CONFIDENCE_THRESHOLD
    ):
        Reset_Gesture_Candidate(hand_state)
        return None

    last_time = hand_state["last_frame_time"]
    if last_time is not None:
        gap = float(frame_timestamp) - float(last_time)
        if gap < 0 or gap > GESTURE_MAX_FRAME_GAP_SEC:
            Reset_Gesture_Candidate(hand_state)
    hand_state["last_frame_time"] = float(frame_timestamp)

    if hand_state["candidate_label"] == label:
        hand_state["stable_count"] += 1
        hand_state["confidence_sum"] += float(confidence)
    else:
        hand_state["candidate_label"] = label
        hand_state["stable_count"] = 1
        hand_state["confidence_sum"] = float(confidence)

    if hand_state["stable_count"] < GESTURE_STABLE_FRAMES:
        return None
    if hand_state["locked_label"] == label:
        return None

    mean_confidence = hand_state["confidence_sum"] / hand_state["stable_count"]
    hand_state["locked_label"] = label
    return {
        "hand": event["hand"],
        "label": label,
        "confidence": mean_confidence,
        "timestamp": float(frame_timestamp),
    }


def Confirm_Gestures_In_Frame(machine, raw_events, frame_timestamp):
    if isinstance(frame_timestamp, bool) or not isinstance(frame_timestamp, (int, float)):
        raise ValueError("frame_timestamp must be numeric")
    if not isinstance(raw_events, (list, tuple)):
        raw_events = []

    event_by_hand = {}
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        hand = Normalize_Hand(raw_event.get("hand"))
        if hand is None:
            continue
        event = {
            "hand": hand,
            "label": Normalize_Gesture_Label(hand, raw_event.get("label")),
            "confidence": raw_event.get("confidence"),
        }
        previous = event_by_hand.get(hand)
        new_score = event["confidence"] if isinstance(event["confidence"], (int, float)) else -1
        old_confidence = previous.get("confidence") if previous else None
        old_score = old_confidence if isinstance(old_confidence, (int, float)) else -1
        if previous is None or new_score > old_score:
            event_by_hand[hand] = event

    confirmed = []
    for hand in VALID_HANDS:
        hand_state = machine["gesture_debounce"][hand]
        event = event_by_hand.get(hand)
        if event is None:
            Mark_Hand_Missing(hand_state)
            continue
        result = Debounce_One_Hand(hand_state, event, frame_timestamp)
        if result is not None:
            confirmed.append(result)
    return confirmed


def Command_Signature(event):
    return (
        event.get("command_type"),
        event.get("mode"),
        event.get("target"),
        event.get("action"),
    )


def Finalize_Command(machine, command_event, refresh_selection=True):
    signature = Command_Signature(command_event)
    timestamp = command_event["timestamp"]
    last_signature = machine.get("last_command_signature")
    last_time = machine.get("last_command_time")
    if signature == last_signature and isinstance(last_time, (int, float)):
        elapsed = timestamp - float(last_time)
        if 0 <= elapsed < COMMAND_DEDUP_SEC:
            return None

    machine["last_command_signature"] = signature
    machine["last_command_time"] = timestamp
    if refresh_selection:
        selection = machine["current_selection"]
        if selection.get("mode") and selection.get("target"):
            selection["last_updated"] = timestamp
    return command_event


def Handle_Confirmed_Gesture(machine, gesture_event):
    hand = Normalize_Hand(gesture_event.get("hand"))
    label = Normalize_Gesture_Label(hand, gesture_event.get("label")) if hand else None
    timestamp = gesture_event.get("timestamp")
    if (
        hand is None
        or label is None
        or isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
    ):
        return None
    timestamp = float(timestamp)

    if hand == "Right" and label == PANIC_GESTURE:
        command = Make_Command_Event(
            "PANIC_STOP", None, "panic", None, "gesture_panic", timestamp
        )
        Reset_Selection(machine)
        return Finalize_Command(machine, command, refresh_selection=False)

    expired = Expire_Selection_If_Needed(machine, timestamp)
    selection = machine["current_selection"]
    if hand == "Left":
        selection["mode"] = MODE_BY_LEFT_GESTURE[label]
        selection["target"] = None
        selection["last_updated"] = timestamp
        return Make_Selection_Event(selection, timestamp, f"left_{label}")

    if expired:
        return Make_Rejected_Event("gesture", "selection_timeout", timestamp)
    mode = selection.get("mode")
    if mode is None:
        return Make_Rejected_Event("gesture", "mode_not_selected", timestamp)
    target = TARGET_BY_MODE[mode].get(label)
    if target is None:
        return Make_Rejected_Event("gesture", "target_not_supported_in_mode", timestamp)
    selection["target"] = target
    selection["last_updated"] = timestamp
    return Make_Selection_Event(selection, timestamp, f"right_{label}")


def Process_Gesture_Frame(machine, raw_events, frame_timestamp):
    outputs = []
    for event in Confirm_Gestures_In_Frame(machine, raw_events, frame_timestamp):
        output = Handle_Confirmed_Gesture(machine, event)
        if output is not None:
            outputs.append(output)
    return outputs


def Normalize_Voice_Event(voice_event):
    if not isinstance(voice_event, dict):
        return None
    action = voice_event.get("action")
    confidence = voice_event.get("confidence")
    timestamp = voice_event.get("timestamp")
    if not isinstance(action, str):
        return None
    action = action.strip().upper()
    if action not in KWS_ACTIONS:
        return None
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return None
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return None
    return {
        "action": action,
        "confidence": float(confidence),
        "timestamp": float(timestamp),
    }


def Process_Voice_Event(machine, voice_event):
    event = Normalize_Voice_Event(voice_event)
    if event is None:
        return None
    timestamp = event["timestamp"]
    if Expire_Selection_If_Needed(machine, timestamp):
        return Make_Rejected_Event("voice", "selection_timeout", timestamp)

    selection = machine["current_selection"]
    mode = selection.get("mode")
    target = selection.get("target")
    if mode is None:
        return Make_Rejected_Event("voice", "mode_not_selected", timestamp)
    if target is None:
        return Make_Rejected_Event("voice", "target_not_selected", timestamp)

    supported_actions = CAPABILITIES.get(mode, {}).get(target)
    if supported_actions is None:
        return Make_Rejected_Event("voice", "target_not_supported_in_mode", timestamp)
    if event["action"] not in supported_actions:
        return Make_Rejected_Event("voice", "action_not_supported", timestamp)

    command = Make_Command_Event(
        "DESKTOP_ACTION",
        mode,
        target,
        event["action"],
        "gesture_voice",
        timestamp,
    )
    return Finalize_Command(machine, command)
