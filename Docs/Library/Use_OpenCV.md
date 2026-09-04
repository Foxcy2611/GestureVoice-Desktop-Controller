# Hướng dẫn cơ bản sử dụng thư viện OpenCV (Python)

Tài liệu này tổng hợp các thao tác cốt lõi nhất dựa trên các thư viện mẫu từ repository chính thức của OpenCV. OpenCV mặc định sử dụng hệ màu **BGR** (Blue, Green, Red) thay vì RGB thông thường.

## 1. Khởi tạo và xử lý luồng Video / Webcam

Để thu thập dữ liệu hình ảnh liên tục theo thời gian thực, `cv2.VideoCapture` là class quan trọng nhất được sử dụng để mở và đọc luồng dữ liệu từ camera.

```python
import cv2

# Khởi tạo luồng video từ webcam (số 0 thường là camera mặc định của máy)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Đọc từng khung hình (frame) từ camera
    # Bool, numpy array
    success, frame = cap.read()
    if not success:
        print("Không thể đọc dữ liệu từ webcam.")
        break

    # Lật ngược ảnh trông dễ nhìn như soi gương
    frame = cv2.flip(frame, 1)
    # Hiển thị khung hình lên cửa sổ có tên "Webcam"
    cv2.imshow("Webcam", frame)

    # Đợi 1 mili-giây, nếu người dùng nhấn phím 'q' (mã ASCII) thì thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên phần cứng và đóng toàn bộ cửa sổ
cap.release()
cv2.destroyAllWindows()
```

## 2. Các hàm vẽ đồ họa cơ bản (Drawing Functions)

Các hàm này can thiệp trực tiếp vào ma trận điểm ảnh để vẽ các hình học cơ bản. Thường được sử dụng để vẽ bounding box, các điểm landmark hoặc thông số lên màn hình.

### Vẽ đường thẳng (Line)
```python
# Cú pháp: cv2.line(ảnh, điểm_đầu, điểm_cuối, màu_sắc, độ_dày)
cv2.line(frame, (0, 0), (100, 100), (0, 255, 0), 2) # Vẽ đường màu xanh lá
```

### Vẽ hình tròn (Circle)
```python
# Cú pháp: cv2.circle(ảnh, tọa_độ_tâm, bán_kính, màu_sắc, độ_dày)
# Lưu ý: Nếu độ dày = -1, hình tròn sẽ được tô đặc (filled)
cv2.circle(frame, (50, 50), 5, (0, 0, 255), -1) 
# Vẽ một chấm đỏ đặc
```

### Vẽ hình chữ nhật (Rectangle)
```python
# Cú pháp: cv2.rectangle(ảnh, góc_trên_trái, góc_dưới_phải, màu_sắc, độ_dày)
cv2.rectangle(frame, (10, 10), (200, 200), (255, 0, 0), 2) # Vẽ viền xanh dương
```

### Ghi văn bản (Text)
```python
# Cú pháp: cv2.putText(ảnh, nội_dung, tọa_độ_góc_dưới_trái, font_chữ, tỷ_lệ_font, màu_sắc, độ_dày)
cv2.putText(frame, "Hello OpenCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```

## 3. Chuyển đổi không gian màu (Color Space Conversion)

Do OpenCV đọc ảnh dưới dạng BGR, nhưng phần lớn các thư viện AI hiện đại hoặc thư viện vẽ đồ thị (như Matplotlib) lại yêu cầu đầu vào là RGB, thao tác chuyển đổi màu là bắt buộc.

```python
# Chuyển đổi từ BGR (mặc định của OpenCV) sang RGB
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Chuyển đổi từ BGR sang ảnh xám (Grayscale) để giảm khối lượng tính toán
image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

## 4. Xử lý ảnh tĩnh (Image I/O)

Các thao tác cơ bản khi cần đọc hoặc lưu ma trận điểm ảnh thành các tệp tin lưu trữ trên ổ cứng (như .jpg, .png).

```python
# Đọc một bức ảnh tĩnh từ ổ cứng vào biến
img = cv2.imread("duong_dan_toi_file_anh.jpg")

# Lưu ma trận ảnh hiện tại thành một file mới
cv2.imwrite("ket_qua_xu_ly.png", img)
```