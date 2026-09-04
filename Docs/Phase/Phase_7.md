# Phase 7 — System Integration

## Mục tiêu

Tạo một entry point duy nhất nối webcam, micro, hai model, State Machine và
Desktop Executor.

```mermaid
sequenceDiagram
    actor User
    participant Camera
    participant GestureAI
    participant StateMachine
    participant KWS
    participant Executor
    participant App

    User->>Camera: left gesture
    Camera->>GestureAI: 5 stable frames
    GestureAI->>StateMachine: select mode
    User->>Camera: right gesture
    GestureAI->>StateMachine: select target
    User->>KWS: ON/OFF/UP/DOWN
    KWS->>StateMachine: VoiceEvent
    StateMachine->>Executor: DesktopCommand
    Executor->>App: hotkey / Windows API
    App-->>User: visible action
```

## Entry point

```powershell
# Chỉ kiểm tra model
.\venv\Scripts\python.exe .\P7_System_Integration\Desktop_Runtime.py --check-models

# Nhận diện thật, không thao tác hệ điều hành
.\venv\Scripts\python.exe .\P7_System_Integration\Desktop_Runtime.py

# Nhận diện và thao tác thật
.\venv\Scripts\python.exe .\P7_System_Integration\Desktop_Runtime.py --execute
```

Runtime đo noise nền trong hai giây, giữ energy gate, VAD, spectral denoise,
voting và cooldown của Phase 4. OpenCV chỉ hiển thị preview/debug, không phải
dashboard.

## Kiểm thử tích hợp

`Test_System_Integration.py` phát event giả từ gesture đến executor dry-run.
Kết quả hiện tại: `5/5 PASS`.
