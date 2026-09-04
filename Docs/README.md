# Tài liệu GestureVoice Desktop Controller

## Theo Phase

- [Phase 1](Phase/Phase_1.md): Data Collection.
- [Phase 2](Phase/Phase_2.md): Data Preprocessing.
- [Phase 3](Phase/Phase_3.md): Model Training.
- [Phase 4](Phase/Phase_4.md): Real-time Evaluation.
- [Phase 5](Phase/Phase_5.md): Desktop State Machine.
- [Phase 6](Phase/Phase_6.md): Desktop Action Executor.
- [Phase 7](Phase/Phase_7.md): System Integration.

## Thiết kế và vận hành

- [Gesture + KWS Mapping](Design/Gesture_KWS_Mapping.md).
- [Keyword Policy](Design/Keyword_Policy.md).
- [Runtime Architecture](Workflow/Desktop_Runtime.md).
- [Test Plan](Testing/Test_Plan.md).

```mermaid
flowchart TB
    P1[Phase 1<br/>Collect] --> P2[Phase 2<br/>Preprocess]
    P2 --> P3[Phase 3<br/>Train]
    P3 --> P4[Phase 4<br/>Real-time inference]
    P4 --> P5[Phase 5<br/>State machine]
    P5 --> P6[Phase 6<br/>Desktop executor]
    P6 --> P7[Phase 7<br/>Integration & evaluation]
```
