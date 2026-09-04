# Sơ lược về thư viện MediaPipe (Python)

Tất cả lý thuyết trên được lấy từ đây [Import MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/)

## ✋ Khởi tạo MediaPipe Hands (Hands)

Để sử dụng MediaPipe Hands, ta khởi tạo đối tượng với hàm `__init__`:

```python
def __init__(
    self,
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
):
```

### 1. static_image_mode

* **False:** Xử lý như video stream liên tục, tối ưu cho webcam
* **True:** Xử lý theo ảnh tĩnh riêng lẻ

### 2. max_num_hands (int)

* Số lượng bàn tay tối đa cần phát hiện trong một frame.
* Mặc định: 2 (tay trái + tay phải).

### 3. model_complexity (int)

* Độ phức tạp của model landmark: 0 hoặc 1.
* Giá trị cao hơn → chính xác hơn nhưng chậm hơn.

### 4. min_detection_confidence (float)

* Ngưỡng tin cậy tối thiểu (0.0–1.0) để coi việc phát hiện bàn tay là thành công.
* Ví dụ: 0.5 nghĩa là chỉ chấp nhận khi confidence ≥ 50%.

### 5. min_tracking_confidence (float)

* Ngưỡng tin cậy tối thiểu (0.0–1.0) để coi landmark bàn tay được theo dõi thành công qua nhiều frame.
* Nếu thấp quá thì dễ mất tracking khi tay di chuyển nhanh.

## 🖌️ Bộ vẽ (Drawing Utils)

Đóng vai trò như "cây cọ vẽ", làm nhiệm vụ hiển thị trực quan các dữ liệu toạ độ lên hình ảnh. Hàm trung tâm được sử dụng nhiều nhất là draw_landmarks().

### 1. Chuyển đổi hệ tọa độ (_normalized_to_pixel_coordinates)

* Mô hình AI chỉ trả về toạ độ x, y ở dạng tỷ lệ tương đối (từ 0.0 đến 1.0).
* Hàm này làm nhiệm vụ nhân tỷ lệ đó với kích thước thực tế của khung hình (chiều rộng, chiều cao) để tính ra toạ độ pixel chuẩn xác trên màn hình.

### 2. Cơ chế lọc điểm nhiễu (Thresholds)

* Tích hợp sẵn các ngưỡng `_VISIBILITY_THRESHOLD` và `_PRESENCE_THRESHOLD` (mặc định là 0.5).
* Nếu một bàn tay bị che khuất hoặc AI nhận diện một điểm mốc với độ tin cậy dưới 50%, điểm đó sẽ lập tức bị bỏ qua và không được vẽ lên hình.

### 3. draw_landmarks

* Vẽ xương (cv2.line): Duyệt qua danh sách các cặp điểm cần kết nối (ví dụ điểm 0 nối với 1) để kẻ các đoạn thẳng.
* Vẽ khớp (cv2.circle): Vẽ các điểm chốt đè lên trên đường nối để tối ưu thẩm mỹ. Mỗi điểm thường được vẽ hai lần: một viền trắng to bên ngoài và một lõi màu (lấy từ bảng màu) bên trong.

## 🎨 Bộ quy chuẩn thẩm mỹ (Drawing Styles)

Đóng vai trò như một "bảng màu" (palette). Nó không trực tiếp vẽ bất cứ thứ gì lên hình, mà chỉ chứa các hàm trả về cấu hình (preset) về màu sắc, độ dày nét vẽ, và kích thước chấm tròn.

### 1. get_default_hand_landmarks_style(): 

* Hàm này trả về một dictionary (từ điển) quy định màu sắc cho từng điểm mốc (landmark) trong số 21 điểm của bàn tay. 
* Nó sử dụng class DrawingSpec để định nghĩa mã màu hệ BGR, độ dày (thickness), và bán kính vòng tròn (circle_radius). 
* Nhờ hàm này mà khi vẽ, ngón cái, ngón trỏ, ngón út tự động có các màu khác nhau để dễ phân biệt.

### 2. get_default_hand_connections_style(): 

* Hàm này định nghĩa DrawingSpec cho các đường thẳng (connections) nối giữa các khớp xương 
* Ví dụ: đường nối từ điểm 0 đến điểm 1 có màu gì, độ dày bao nhiêu.

## 🧮 Hiển thị thông tin lên webcam

Khi gọi `hands.process`, thì bạn có thể gọi thêm các trường sau đây của nó

### 1. multi_hand_landmarks

* Chứa các điểm mốc trên mỗi bàn tay được phát hiện.

### 2. multi_hand_world_landmarks

* Chứa các điểm mốc trên mỗi bàn tay được phát hiện trong tọa độ 3D thực tế tính bằng mét với gốc tọa độ tại tâm hình học gần đúng của bàn tay.

### 3. multi_handedness
* Chứa hướng thuận tay (tay trái so với tay phải) của bàn tay được phát hiện.